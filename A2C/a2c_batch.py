import torch
import torch.nn as nn
import numpy as np
import matplotlib.pylab as plt
import torch.distributions as ptd
import gymnasium as gym
import scipy
""" def discount_cumsum(x: torch.Tensor, discount):
    sum = 0.0
    result = torch.zeros_like(x)
    for i in range(len(x) - 1, -1, -1):
        sum = discount * sum + x[i]
        result[i] = sum
    return result """
def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
#currently only for continuous actions
class Actor(nn.Module):
    def __init__(self, ob_dim, ac_dim, layer_size):
        super().__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim

        self.l1 = nn.Linear(self.ob_dim, layer_size)
        self.ac1 = nn.ReLU()
        self.l2 = nn.Linear(layer_size, layer_size)
        self.ac2 = nn.ReLU()
        self.out = nn.Linear(layer_size, self.ac_dim)
        self.log_std = nn.Parameter(torch.zeros(self.ac_dim))
    def forward(self, state):
        out = self.ac1(self.l1(state))
        out = self.ac2(self.l2(out))
        out = self.out(out)
        std = torch.exp(self.log_std)
        return out, std
    
class Critic(nn.Module):
    def __init__(self, ob_dim, layer_size):
        super().__init__()
        self.ob_dim = ob_dim

        self.l1 = nn.Linear(self.ob_dim, layer_size)
        self.ac1 = nn.ReLU()
        self.l2 = nn.Linear(layer_size, layer_size)
        self.ac2 = nn.ReLU()
        self.l3 = nn.Linear(layer_size, 1)
    def forward(self, state):
        out = self.ac1(self.l1(state))
        out = self.ac2(self.l2(out))
        out = self.l3(out)
        return out.squeeze()

class A2C(nn.Module):
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.gamma = self.config.gamma
        self.lam = self.config.lam
        self.seed = self.config.seed
        self.env.reset(seed=self.seed)
        self.device = 'cpu'
        torch.manual_seed(seed=self.seed)

        self.ob_dim = self.env.observation_space.shape[0]
        self.ac_dim = self.env.action_space.shape[0]
        self.actor = Actor(self.ob_dim, self.ac_dim, self.config.layer_size).to(self.device)
        self.critic = Critic(self.ob_dim, self.config.layer_size).to(self.device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.config.pi_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.config.v_lr)
        
        self.quant_input = torch.ao.quantization.QuantStub()
        self.dequant_output = torch.ao.quantization.DeQuantStub()

    def forward(self, state):
        out = self.quant_input(state)
        out = self.actor(out)[0]
        out = self.dequant_output(out)
        return out
    
    def to(self, device):
        model = super().to(device)
        model.device = device
        return model
    
    def calc_advantage(self, ob, next_ob, rewards, done):
        values = self.critic(torch.as_tensor(ob, dtype=torch.float, device=self.device)).detach().cpu().numpy()
        next_vals = self.critic(torch.as_tensor(next_ob, dtype=torch.float, device=self.device)).detach().cpu().numpy()
        delta = rewards + (1.0 - done) * self.gamma * next_vals - values
        advantages = discount_cumsum(delta, self.lam * self.gamma)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return advantages
    
    def sample_batch(self):
        i = 0
        paths, returns = [], []
        state, _ = self.env.reset()
        while i < self.config.batch_size:
            states, actions, rewards, dones, next_states = [], [], [], [], []
            state, _ = self.env.reset()
            episode_reward = 0
            for step in range(self.config.max_ep_len):
                states.append(state)
                mu, std = self.actor(torch.as_tensor(state, dtype=torch.float, device=self.device))
                dist = ptd.MultivariateNormal(loc=mu, scale_tril=torch.diag(std))
                action = dist.sample().detach().cpu().numpy()
                actions.append(action)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_states.append(next_state)
                rewards.append(reward)
                done = terminated or truncated
                dones.append(done)
                episode_reward += reward
                state = next_state
                i += 1
                if done or step == self.config.max_ep_len - 1:
                    returns.append(episode_reward)
                    break
                if i >= self.config.batch_size:
                    break
            path = {
                'states': np.array(states),
                'actions': np.array(actions),
                'rewards': np.array(rewards),
                'done': np.array(dones),
                'next_states': np.array(next_states)
            }
            paths.append(path)
        return paths, returns
    
    def get_returns(self, paths):
        all_returns = []
        for path in paths:
            rewards = path["rewards"]
            returns = np.zeros(rewards.shape)
            returns[len(rewards) - 1] = rewards[len(rewards) - 1]
            for index, reward in reversed(list(enumerate(rewards))):
                if index < len(rewards) - 1:
                    returns[index] = reward + self.gamma*returns[index+1]
            all_returns.append(returns)
        return np.concatenate(all_returns)
    
    def update_critic(self, returns, states):
        values = self.critic(torch.as_tensor(states, dtype=torch.float, device=self.device))
        returns = torch.as_tensor(returns, dtype=torch.float, device=self.device)
        loss_critic = torch.nn.functional.mse_loss(values, returns)
        self.critic_optim.zero_grad()
        loss_critic.backward()
        self.critic_optim.step()

    def update_actor(self, advantages, states, actions):
        mu, std = self.actor(torch.as_tensor(states, dtype=torch.float, device=self.device))
        dist = ptd.MultivariateNormal(loc=mu, scale_tril=torch.diag(std))
        log_probs = dist.log_prob(torch.as_tensor(actions, dtype=torch.float, device=self.device))
        loss_actor = -torch.sum(log_probs * torch.as_tensor(advantages, dtype=torch.float, device=self.device))
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()

    def evaluation(self, seed = None):
        env = gym.make(self.config.env)
        ep_reward = 0
        if seed is None:
            seed = self.config.seed
        state, _ = env.reset(seed = seed + 100)
        for i in range(self.config.eval_epochs):
            state, _ = env.reset()
            done = False
            while not done:
              action = self(torch.as_tensor(state, dtype=torch.float, device=self.device)).detach().cpu().numpy()
              state, reward, terminated, truncated, _ = env.step(action)
              ep_reward += reward
              done = terminated or truncated
        print("---------------------------------------")
        print(f"Evaluation over {self.config.eval_epochs} episodes: {ep_reward/self.config.eval_epochs:.3f}")
        print("---------------------------------------")

    

    def train_agent(self):
        best_avg = None
        for ep in range(self.config.num_epoch):
            paths, episodic_rewards = self.sample_batch()
            states = np.concatenate([path["states"] for path in paths])
            actions = np.concatenate([path["actions"] for path in paths])
            rewards = np.concatenate([path["rewards"] for path in paths])
            next_states = np.concatenate([path["next_states"] for path in paths])
            done = np.concatenate([path["done"] for path in paths])
            returns = self.get_returns(paths)

            self.update_critic(returns, states)
            advantages = self.calc_advantage(states, next_states, rewards, done)
            self.update_actor(advantages, states, actions)
            avg_reward = np.mean(episodic_rewards)
            if best_avg is None or best_avg <= avg_reward:
                best_avg = avg_reward
                self.save_model(f"models/a2c-{self.config.env_name}-seed-{self.seed}-best.pt")
            print(f"Iter {ep}: Avg reward:{avg_reward:.2f}")
            if (ep + 1) % 5 == 0:
                self.evaluation()
                self.save_model(f"models/a2c-{self.config.env_name}-seed-{self.seed}.pt")
        print("Best Avg reward: {:.2f}".format(best_avg))
                
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
