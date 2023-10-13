import torch
import torch.nn as nn
import torch.distributions as ptd
import numpy as np
import gymnasium as gym
import scipy

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
class ValueNetwork(nn.Module):
    def __init__(self, ob_size, gamma, lamb, layer_size):
        super().__init__()
        self.gamma = gamma
        self.lam = lamb
        self.l1 = nn.Linear(ob_size, layer_size)
        self.ac1 = nn.ReLU()
        self.l2 = nn.Linear(layer_size, layer_size)
        self.ac2 = nn.ReLU()
        self.l3 = nn.Linear(layer_size, 1)
    def forward(self, x):
        out = self.ac1(self.l1(x))
        out = self.ac2(self.l2(out))
        return self.l3(out).squeeze()
    def calc_advantage(self, obs, next_obs, rewards):
        values = self(obs).detach().cpu().numpy()
        next_values = self(next_obs).detach().cpu().numpy()
        delta = rewards + self.gamma * next_values - values
        advantages = discount_cumsum(delta, self.lam * self.gamma)
        advantages = (advantages - np.mean(advantages)) / np.std(advantages)
        return advantages
class ActionNetwork(nn.Module):
    def __init__(self, ob_size, act_size, layer_size):
        super().__init__()
        self.l1 = nn.Linear(ob_size, layer_size)
        self.ac1 = nn.ReLU()
        self.l2 = nn.Linear(layer_size, layer_size)
        self.ac2 = nn.ReLU()
        self.l3 = nn.Linear(layer_size, act_size)
    def forward(self, x):
        out = self.ac1(self.l1(x))
        out = self.ac2(self.l2(out))
        return self.l3(out)

class GaussianPolicy(nn.Module):
    def __init__(self, ob_size, act_size, layer_size):
        super().__init__()
        self.action_size = act_size
        self.network = ActionNetwork(ob_size, act_size, layer_size)
    def action_dist(self, x):
        mean = self(x)
        dist = ptd.MultivariateNormal(loc=mean, scale_tril=torch.eye(self.action_size, device=mean.device))
        return dist
    def forward(self, x):
        mean = self.network(x)
        return mean
class CategoricalPolicy(nn.Module):
    def __init__(self, ob_size, act_size):
        super().__init__()
        self.action_size = act_size
        self.network = ActionNetwork(ob_size, act_size)
    def action_dist(self, x):
        logits = self(x)
        dist = ptd.Categorical(logits=logits)
        return dist
    def forward(self, x):
        logits = self.network(x)
        return logits

class PPO(nn.Module):
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.seed = self.config.seed
        torch.manual_seed(seed=self.seed)
        self.env.reset(seed=self.seed)

        self.observation_size = self.env.observation_space.shape[0]
        self.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.action_size = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]

        self.gamma = self.config.gamma
        self.lam = self.config.lam
        self.clip = self.config.clip
        self.v_lr = self.config.v_lr
        self.pi_lr = self.config.pi_lr
        self.device = 'cpu'
        self.baseline = ValueNetwork(self.observation_size, self.gamma, self.lam, self.config.layer_size).to(self.device)
        self.policy = CategoricalPolicy(self.observation_size, self.action_size, self.config.layer_size).to(self.device) if self.discrete else GaussianPolicy(self.observation_size, self.action_size, self.config.layer_size).to(self.device)
        
        self.opt_baseline = torch.optim.Adam(self.baseline.parameters(), lr=self.v_lr)
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=self.pi_lr)
        self.quant_input = torch.ao.quantization.QuantStub()
        self.dequant_output = torch.ao.quantization.DeQuantStub()

    def to(self, device):
        model = super().to(device)
        model.device = device
        return model

    def sample_batch(self):
        i = 0
        paths, returns = [], []
        state, _ = self.env.reset()
        while i < self.config.batch_size:
            states, actions, rewards, log_probs, next_states = [], [], [], [], []
            state, _ = self.env.reset()
            episode_reward = 0
            for step in range(self.config.max_ep_len):
                states.append(state)
                dist = self.policy.action_dist(torch.as_tensor(state, dtype=torch.float, device=self.device))
                action = dist.sample()
                log_prob = dist.log_prob(action)
                action = action.detach().cpu().numpy()
                log_prob = log_prob.detach().cpu().numpy()
                actions.append(action)
                log_probs.append(log_prob)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_states.append(next_state)
                rewards.append(reward)
                done = terminated or truncated
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
                'log_probs': np.array(log_probs),
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
    
    def update_policy(self, states, actions, old_log_probs, advantages):
        advantages = torch.as_tensor(advantages, dtype=torch.float, device=self.device)
        dist = self.policy.action_dist(torch.as_tensor(states, dtype=torch.float, device=self.device))
        actions = torch.as_tensor(actions, dtype=torch.float, device=self.device)
        log_probs = dist.log_prob(actions)
        r_theta = torch.exp(log_probs - torch.as_tensor(old_log_probs, dtype=torch.float, device=self.device))
        clip_r_theta = torch.clip(r_theta, 1.0 - self.clip, 1.0 + self.clip)
        loss = -(torch.min(torch.mul(r_theta, advantages), torch.mul(clip_r_theta, advantages))).mean()
        self.opt_policy.zero_grad()
        loss.backward()
        self.opt_policy.step()
    def update_baseline(self, returns, states):
        values = self.baseline(torch.as_tensor(states, dtype=torch.float, device=self.device))
        returns = torch.as_tensor(returns, dtype=torch.float, device=self.device)
        loss = torch.nn.functional.mse_loss(returns, values)
        self.opt_baseline.zero_grad()
        loss.backward()
        self.opt_baseline.step()
    
    def train_agent(self):
        best_avg = None
        for ep in range(self.config.epoch):
            paths, episodic_rewards = self.sample_batch()
            states = np.concatenate([path["states"] for path in paths])
            actions = np.concatenate([path["actions"] for path in paths])
            rewards = np.concatenate([path["rewards"] for path in paths])
            next_states = np.concatenate([path["next_states"] for path in paths])
            old_logprobs = np.concatenate([path["log_probs"] for path in paths])
            returns = self.get_returns(paths)

            states = torch.as_tensor(states, dtype=torch.float, device=self.device)
            next_states = torch.as_tensor(next_states, dtype=torch.float, device=self.device)
            advantages = self.baseline.calc_advantage(states, next_states, rewards)
            for _ in range(self.config.update_freq):
                self.update_policy(states, actions, old_logprobs, advantages)
                self.update_baseline(returns, states)
            avg_reward = np.mean(episodic_rewards)

            if best_avg is None or best_avg <= avg_reward:
                best_avg = avg_reward
                self.save_model(f"./models/ppo-{self.config.env_name}-seed-{self.seed}-best.pt")
            self.save_model(f"./models/ppo-{self.config.env_name}-seed-{self.seed}.pt")
            print("Iter {}: Avg reward:{:.2f}".format(ep, avg_reward))
        print("Best Avg reward: {:.2f}".format(best_avg))

    def forward(self, ob):
        ob = self.quant_input(ob)
        out = None
        if self.discrete:
            logits = self.policy(ob)
            probs = torch.nn.functional.softmax(logits, dim=0)
            out = torch.argmax(probs)
        else:
            out = self.policy(ob)
        out = self.dequant_output(out)
        return out
    def evaluation(self, seed=None):
        returns = []
        if seed is None:
            seed = self.seed
        for ep in range(10):
            state, _ = self.env.reset()
            done = False
            ep_reward = 0
            while not done:
                action = self(torch.as_tensor(state, dtype=torch.float, device=self.device)).detach().cpu().numpy()
                state, reward, terminated, truncated, _ = self.env.step(action)
                ep_reward += reward
                done = terminated or truncated
            returns.append(ep_reward)
        print("Avg reward: {:.2f}".format(np.mean(returns)))

    def save_model(self, path):
        torch.save(self.state_dict(), path)
    def load_model(self, path):
        self.load_state_dict(torch.load(path))

