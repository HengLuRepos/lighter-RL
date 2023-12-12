import torch
import torch.nn as nn
import numpy as np
import torch.distributions as ptd
import gymnasium as gym
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def np2torch(np_arr):
    np_arr = torch.from_numpy(np_arr) if isinstance(np_arr,np.ndarray) else np_arr
    return np_arr.to(device).float()
#currently only for continuous actions
class Actor(nn.Module):
    def __init__(self, ob_dim, ac_dim):
        super().__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim

        self.l1 = nn.Linear(self.ob_dim, 256)
        self.ac1 = nn.ReLU()
        self.l2 = nn.Linear(256, 256)
        self.ac2 = nn.ReLU()
        self.mu = nn.Linear(256, ac_dim)
        self.log_std = nn.Parameter(torch.zeros(self.ac_dim, dtype=torch.float)).to(device)
    
    def forward(self, state):
        out = self.ac1(self.l1(state))
        out = self.ac2(self.l2(out))
        mu = self.mu(out)
        #log_std = self.log_std(out)
        std = torch.exp(self.log_std)
        return mu, std

class Critic(nn.Module):
    def __init__(self, ob_dim):
        super().__init__()
        self.ob_dim = ob_dim

        self.l1 = nn.Linear(self.ob_dim, 256)
        self.ac1 = nn.ReLU()
        self.l2 = nn.Linear(256, 256)
        self.ac2 = nn.ReLU()
        self.l3 = nn.Linear(256, 1)
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
        self.seed = self.config.seed
        self.env.reset(seed=self.seed)
        torch.manual_seed(seed=self.seed)

        self.ob_dim = self.env.observation_space.shape[0]
        self.ac_dim = self.env.action_space.shape[0]
        self.actor = Actor(self.ob_dim, self.ac_dim).to(device)
        self.critic = Critic(self.ob_dim).to(device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.config.pi_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.config.v_lr)

    def forward(self, state):
        return self.actor(state)[0]
    
    def evaluation(self):
        env = gym.make(self.config.env)
        ep_reward = 0
        state, _ = env.reset(seed = self.config.seed + 100)
        for i in range(self.config.eval_epochs):
            state, _ = env.reset()
            done = False
            while not done:
              action = self(np2torch(state)).detach().cpu().numpy()
              state, reward, terminated, truncated, _ = env.step(action)
              ep_reward += reward
              done = terminated or truncated
        print("---------------------------------------")
        print(f"Evaluation over {self.config.eval_epochs} episodes: {ep_reward/self.config.eval_epochs:.3f}")
        print("---------------------------------------")
        return ep_reward

    def train_agent(self):
        state, _ = self.env.reset()
        done = False
        all_ep_rewards = []
        ep_reward = 0
        best_reward = None
        for t in range(self.config.max_timestamp):
            mu, std = self.actor(np2torch(state))
            dist = ptd.MultivariateNormal(loc=mu, scale_tril=torch.diag(std))
            action = dist.sample()
            #noise = torch.randn(std.shape, device=device)
            #action = torch.tanh(mu + torch.mul(std, noise)) * self.config.a_high
            next_state, reward, terminated, truncated, info = self.env.step(action.detach().cpu().numpy())
            ep_reward += reward
            done = terminated or truncated
            advantage = reward + self.critic(np2torch(next_state)) * (1.0 - done) * self.gamma - self.critic(np2torch(state))
            state = next_state
            
            #update critic
            loss_critic = torch.pow(advantage, 2).mean()
            self.critic_optim.zero_grad()
            loss_critic.backward()
            self.critic_optim.step()
            
            #update actor
            #log_prob = -torch.log(std) - torch.log(2 * torch.tensor(torch.pi)).to(device) / 2 - ((action - mu)**2)/(2*std**2)
            #log_prob -= (2*(np.log(2) - action - torch.nn.functional.softplus(-2*action))).sum()
            log_prob = dist.log_prob(action)
            loss_actor = -torch.sum(torch.mul(log_prob, advantage.detach()))
            self.actor_optim.zero_grad()
            loss_actor.backward()
            self.actor_optim.step()

            if done:
                state, _ = self.env.reset()
                done = False
                all_ep_rewards.append(ep_reward)
                print(f"Episode {len(all_ep_rewards)} return: {ep_reward:.3f}")
                ep_reward = 0
            if (t + 1) % self.config.eval_freq == 0:
                rw = self.evaluation()
                self.save_model(f"models/a2c-{self.config.env}-seed-{self.seed}.pt")
                if best_reward is None or rw >= best_reward:
                    self.save_model(f"models/a2c-{self.config.env}-seed-{self.seed}-best.pt")
                
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))
            
