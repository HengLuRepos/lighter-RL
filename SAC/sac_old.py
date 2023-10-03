import gymnasium as gym
import torch
import torch.nn as nn
import torch.distributions as ptd
import numpy as np
from collections import deque
import random
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, layer_size):
        super().__init__()
        self.l1 = nn.Linear(input_dim, layer_size)
        self.ac1 = nn.ReLU()
        self.l2 = nn.Linear(layer_size, layer_size)
        self.ac2 = nn.ReLU()
        self.out = nn.Linear(layer_size, output_dim)

        self.init_weights()
    def forward(self, x):
        out = self.ac1(self.l1(x))
        out = self.ac2(self.l2(out))
        out = self.out(out)
        return out
    def init_weights(self):
        nn.init.xavier_normal_(self.l1.weight)
        nn.init.xavier_normal_(self.l2.weight)
        nn.init.xavier_normal_(self.out.weight)
class ReGaussianPolicy(nn.Module):
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.log_alpha = nn.Parameter(torch.tensor(-1.0), requires_grad=True)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.mu_network = MLP(self.observation_dim, self.action_dim, self.config.layer_size)
        self.log_network = MLP(self.observation_dim, self.action_dim, self.config.layer_size)
        self.action_scale = torch.FloatTensor((self.env.action_space.high -
                                               self.env.action_space.low) / 2.)
        self.action_bias = torch.FloatTensor((self.env.action_space.high +
                                              self.env.action_space.low) / 2.)
        self.optimizer = torch.optim.Adam([
            {'params': self.mu_network.parameters()},
            {'params': self.log_network.parameters()}
        ], lr=self.config.pi_lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.config.pi_lr)
    def forward(self, obs, require_logprob=True):
        mean = self.mu_network(obs)
        log_std = self.log_network(obs)
        std = torch.exp(log_std)
        pi_dist = torch.distributions.Normal(mean,std)
        pi_act = pi_dist.rsample()
        logp_pi = None
        if require_logprob:
            logp_pi = pi_dist.log_prob(pi_act).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_act - torch.nn.functional.softplus(-2*pi_act))).sum(axis=1)
        return pi_act, logp_pi, torch.tanh(mean)
    
    def update_actor(self, q_new, log_probs):
        loss = torch.mean(self.log_alpha.exp().detach() * log_probs - q_new)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_alpha = torch.mean(self.log_alpha.exp() * (self.action_dim - log_probs.detach()))
        self.alpha_opt.zero_grad()
        loss_alpha.backward()
        self.alpha_opt.step()

    def reset_optimizer(self):
        self.optimizer = torch.optim.Adam([
            {'params': self.mu_network.parameters()},
            {'params': self.log_network.parameters()}
        ], lr=self.config.pi_lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.config.pi_lr)
        

class SoftQNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim, config):
        super().__init__()
        self.config = config
        self.network = MLP(observation_dim + action_dim, 1, self.config.layer_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.q_lr)

    def update_q_network(self, observations, actions, q_targets):
        inputs = self(observations, actions)
        loss = torch.nn.functional.mse_loss(inputs.squeeze(), q_targets.float().detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def forward(self, obs, actions):
        # work for batched version.
        x = torch.cat((obs, actions), 1)
        out = self.network(x)
        return out
    
    def copy(self, q_network):
        with torch.no_grad():
            for param1, param2 in zip(self.parameters(), q_network.parameters()):
                param1.copy_(param2)
    
    def reset_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.q_lr)

class SoftCritic(nn.Module):
    def __init__(self, observation_dim, config):
        super().__init__()
        self.config = config
        self.observation_dim = observation_dim
        self.network = MLP(observation_dim, 1, self.config.layer_size)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.v_lr)
        self.tau = self.config.tau

    def update_critic(self, states, q1, q2, log_probs, alpha):
        log_probs = log_probs.unsqueeze(1)
        q_min = torch.min(q1, q2).detach()
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
        out = self.network(states)
        out = out.view(out.size(0), -1)
        return out
    
    def copy(self, q_network):
        with torch.no_grad():
            for param1, param2 in zip(self.parameters(), q_network.parameters()):
                param1.copy_(param2)
    def reset_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.q_lr)

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
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.seed = self.config.seed
        self.device = 'cpu'
        self.env.reset(seed=self.seed)
        torch.manual_seed(seed=self.seed)
        random.seed(self.seed)

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.actor = ReGaussianPolicy(self.env, self.config).to(self.device)
        self.critic = SoftCritic(self.observation_dim, self.config).to(self.device)
        self.critic_target = SoftCritic(self.observation_dim, self.config).to(self.device)
        self.critic_target.copy(self.critic)
        self.q1 = SoftQNetwork(self.observation_dim, self.action_dim, self.config).to(self.device)
        self.q2 = SoftQNetwork(self.observation_dim, self.action_dim, self.config).to(self.device)
        self.q2.copy(self.q1)
        self.buffer = ReplayBuffer(self.config)


        self.gamma = self.config.gamma

    def _reset_optimizer(self):
        for p in list(self.modules())[1:]:
            if hasattr(p,'reset_optimizer'):
                p.reset_optimizer()
    def to(self, device):
        model = super().to(device)
        model.device = device
        model._reset_optimizer()
        return model
    
    def forward(self, state, deterministic=False):
        action, _, action_d = self.actor(state, require_logprob=False)
        if deterministic:
            return action_d
        return action

    def train_agent(self):
        all_ep_rewards = []
        best_eval = None
        ep_reward = 0
        for i in range(self.config.total_timesteps):
            state, _ = self.env.reset()
            if i < self.config.start_steps:
                action = self.env.action_space.sample()
            else:
                action = self(torch.as_tensor(state, dtype=torch.float, device=self.device)).detach().cpu().numpy()
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            ep_reward += reward
            self.buffer.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                all_ep_rewards.append(ep_reward)
                ep_reward = 0
                state, _ = self.env.reset()
            if i >= self.config.start_steps:
                states, actions, rewards, next_states, done = self.buffer.sample()
                next_states = torch.as_tensor(next_states, dtype=torch.float, device=self.device)
                q_targets = rewards + self.gamma * (1.0 - done) * self.critic_target(next_states).detach().squeeze().cpu().numpy()
                q_targets = torch.as_tensor(q_targets, dtype=torch.float, device=self.device)
                states = torch.as_tensor(states, dtype=torch.float, device=self.device)
                actions = torch.as_tensor(actions, dtype=torch.float, device=self.device)
                actions_sampled, logprobs_sampled, _ = self.actor(states)
                q1 = self.q1(states, actions_sampled).to(self.device)
                q2 = self.q2(states, actions_sampled).to(self.device)
                
                self.q1.update_q_network(states, actions, q_targets)
                self.q2.update_q_network(states, actions, q_targets)
                logprobs_sampled = torch.as_tensor(logprobs_sampled, dtype=torch.float, device=self.device)
                self.critic.update_critic(states, q1, q2, logprobs_sampled, torch.exp(self.actor.log_alpha).item())

                if i % self.config.policy_freq == 0:
                    r_actions, r_log_probs, _ = self.actor(states)
                    q1_new = self.q1(states, r_actions)
                    self.actor.update_actor(q1_new, r_log_probs)

                self.critic_target.hard_update(self.critic)
            self.save_model(f"models/sac-{self.config.env_name}-seed-{self.seed}.pt")
            
            if i % self.config.eval_freq == 0:
                avg_return, _ = self.evaluation()
                if best_eval is None or avg_return >= best_eval:
                    best_eval = avg_return
                    self.save_model(f"models/sac-{self.config.env_name}-seed-{self.seed}-best.pt")
                print(f"Evaluation: Episodic reward: {avg_return:04.2f}")
    def evaluation(self, seed=None):
        if seed == None:
            seed = self.seed
        steps = 0
        returns = 0
        env = gym.make(self.config.env)
        state, _ = env.reset(seed=seed + 100)
        for i in range(self.config.eval_epochs):
            done = False
            while not done:
                action = self(torch.as_tensor(state, dtype=torch.float, device=self.device), deterministic=True).detach().cpu().numpy()
                next_state, reward, terminated, truncated, _ = env.step(action)
                returns += reward
                state = next_state
                done = terminated or truncated
                steps += 1
            state, _ = env.reset()
        return returns/self.config.eval_epochs, steps/self.config.eval_epochs


    def save_model(self, path):
        torch.save(self.state_dict(), path)
    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    