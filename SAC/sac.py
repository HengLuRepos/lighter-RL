import gymnasium as gym
import torch
import torch.nn as nn
import torch.distributions as ptd
import numpy as np
from utils import device, mlp, np2torch
from collections import deque
import random
class ReGaussianPolicy(nn.Module):
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.log_alpha = nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.mu_network = mlp(self.observation_dim,
                                    self.action_dim,
                                    self.config.layer_size,
                                    self.config.n_layers).to(device)
        self.log_network = mlp(self.observation_dim,
                                     self.action_dim,
                                     self.config.layer_size,
                                     self.config.n_layers).to(device)
        self.action_scale = torch.FloatTensor((self.env.action_space.high -
                                               self.env.action_space.low) / 2.)
        self.action_bias = torch.FloatTensor((self.env.action_space.high +
                                              self.env.action_space.low) / 2.)
        self.optimizer = torch.optim.Adam([
            {'params': self.mu_network.parameters()},
            {'params': self.log_network.parameters()}
        ], lr=self.config.pi_lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.config.pi_lr)
        self.alpha = torch.exp(self.log_alpha).to(device)
    def forward(self, obs, require_logprob=True):
        obs = np2torch(obs)
        mean = self.mu_network(obs).to(device)
        log_std = self.log_network(obs).to(device)
        std = torch.exp(log_std).to(device)
        pi_dist = torch.distributions.Normal(mean,std)
        pi_act = pi_dist.rsample()
        logp_pi = None
        if require_logprob:
            logp_pi = pi_dist.log_prob(pi_act).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_act - torch.nn.functional.softplus(-2*pi_act))).sum(axis=1)
        return pi_act.to(device), logp_pi, torch.tanh(mean)
    
    def update_actor(self, q_new, log_probs):
        loss = torch.mean(self.alpha.detach() * log_probs - q_new)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_alpha = torch.mean(self.alpha * self.action_dim - self.alpha * log_probs.detach())
        self.alpha_opt.zero_grad()
        loss_alpha.backward()
        self.alpha_opt.step()

        self.alpha = torch.exp(self.log_alpha).to(device)
        

class SoftQNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim, config):
        super().__init__()
        self.config = config
        self.network = mlp(observation_dim + action_dim,
                                 1,
                                 self.config.layer_size,
                                 self.config.n_layers).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.q_lr)

    def update_q_network(self, observations, actions, q_targets):
        inputs = self(observations, actions)
        loss = torch.nn.functional.mse_loss(inputs.squeeze(), q_targets.float().detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def forward(self, obs, actions):
        # work for batched version.
        x = None
        if isinstance(actions, np.ndarray):
            x = np.concatenate([obs, actions], axis=1)
        else:
            obs = np2torch(obs)
            x = torch.cat((obs, actions), 1)
        out = self.network(np2torch(x))
        out = out.view(out.size(0), -1)
        return out
    
    def copy(self, q_network):
        with torch.no_grad():
            for param1, param2 in zip(self.parameters(), q_network.parameters()):
                param1.copy_(param2)

class SoftCritic(nn.Module):
    def __init__(self, observation_dim, config):
        super().__init__()
        self.config = config
        self.observation_dim = observation_dim
        self.network = mlp(self.observation_dim,
                                 1,
                                 self.config.layer_size,
                                 self.config.n_layers).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.v_lr)
        self.tau = self.config.tau

    def update_critic(self, states, q1, q2, log_probs, alpha):
        log_probs = np2torch(log_probs).unsqueeze(1)
        q_min = torch.min(q1, q2)
        inputs = self(states)
        loss = torch.nn.functional.mse_loss(inputs, q_min - alpha * log_probs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def hard_update(self, critic):
        # update target network value
        with torch.no_grad():
            for param1, param2 in zip(self.parameters(), critic.parameters()):
                param1.copy_(self.tau * param2 + (1.0 - self.tau) * param1)

    def forward(self, states):
        states = np2torch(states)
        out = self.network(states)
        out = out.view(out.size(0), -1)
        return out
    
    def copy(self, q_network):
        with torch.no_grad():
            for param1, param2 in zip(self.parameters(), q_network.parameters()):
                param1.copy_(param2)

class ReplayBuffer:
    def __init__(self, config):
        self.max_len= config.buffer_size
        self.batch_size = config.batch_size
        self.memory = deque(maxlen=self.max_len)
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for state, action, reward, next_state, done in minibatch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        states = np.stack(states)
        actions = np.stack(actions)
        rewards = np.stack(rewards)
        next_states = np.stack(next_states)
        dones = np.stack(dones)
        return states, actions, rewards, next_states, dones

class SAC(nn.Module):
    def __init__(self, env, config, seed):
        super().__init__()
        self.env = env
        self.config = config
        self.seed = seed

        self.env.reset(seed=self.seed)
        torch.manual_seed(seed=self.seed)
        random.seed(seed)

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.actor = ReGaussianPolicy(self.env, self.config)
        self.critic = SoftCritic(self.observation_dim, self.config)
        self.critic_target = SoftCritic(self.observation_dim, self.config)
        self.critic_target.copy(self.critic)
        self.q1 = SoftQNetwork(self.observation_dim, self.action_dim, self.config)
        self.q2 = SoftQNetwork(self.observation_dim, self.action_dim, self.config)
        self.q2.copy(self.q1)
        self.buffer = ReplayBuffer(self.config)


        self.gamma = self.config.gamma
    def forward(self, state, deterministic=False):
        state = np2torch(state)
        action, _, action_d = self.actor(state, require_logprob=False)
        action = action.detach().cpu().numpy()
        if deterministic:
            return action_d.detach().cpu().numpy()
        return action
    
    def train(self):
        all_ep_rewards = []
        for i in range(self.config.num_iter):
            state, _ = self.env.reset()
            for step in range(self.config.explore_step):
                action = self(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.buffer.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    state, _ = self.env.reset()
            
            for _ in range(self.config.update_gradient_freq):
                states, actions, rewards, next_states, done = self.buffer.sample()
                q_targets = rewards + self.gamma * (1.0 - done) * self.critic_target(next_states).detach().squeeze().cpu().numpy()
                q_targets = np2torch(q_targets)
                actions_sampled, logprobs_sampled, _ = self.actor(states)
                q1 = self.q1(states, actions_sampled).to(device)
                q2 = self.q2(states, actions_sampled).to(device)
                
                self.q1.update_q_network(states, actions, q_targets)
                self.q2.update_q_network(states, actions, q_targets)
                self.critic.update_critic(states, q1, q2, logprobs_sampled, self.actor.alpha.item())

                r_actions, r_log_probs, _ = self.actor(states)
                q1_new = self.q1(states, r_actions)
                self.actor.update_actor(q1_new, r_log_probs)

                self.critic_target.hard_update(self.critic)
            
            state, _ = self.env.reset()
            episodic_reward = 0
            done = False
            while not done:
                action = self(state, deterministic=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episodic_reward = reward + self.gamma * episodic_reward
                state = next_state
                done = terminated or truncated
            all_ep_rewards.append(episodic_reward)
            msg = "[EPISODE {}]: Episodic reward: {:04.2f}".format(i, episodic_reward)
            print(msg)
        print("maximum reward: {:.2f}".format(max(all_ep_rewards)))



    