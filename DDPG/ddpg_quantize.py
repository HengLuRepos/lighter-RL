import torch
import torch.nn as nn
import torch.distributions as ptd
import numpy as np
import gymnasium as gym
import torch.nn.functional as F
from collections import deque
import random
"""
    This implementaion follows the DDPG used in TD3 paper.
"""

class Actor(nn.Module):
    def __init__(self, ob_dim, act_dim, config):
        super().__init__()
        self.config = config
        self.a_low = self.config.a_low
        self.a_high = self.config.a_high
        self.explore_noise = self.config.explore_noise
        self.tau = self.config.tau

        self.l1 = nn.Linear(ob_dim, 256)
        self.ac1 = nn.ReLU()
        self.l2 = nn.Linear(256, 256)
        self.ac2 = nn.ReLU()
        self.l3 = nn.Linear(256, act_dim)
        self.ac3 = nn.Tanh()

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.l1.weight)
        nn.init.xavier_normal_(self.l2.weight)
        nn.init.xavier_normal_(self.l3.weight)

    def forward(self, ob):
        out = self.ac1(self.l1(ob))
        out = self.ac2(self.l2(out))
        out = self.ac3(self.l3(out))
        return out
    
    def explore(self, state):
        action = self(state) * self.config.a_high
        noise = torch.normal(mean=0.0, std=self.explore_noise, size=action.size(), device=action.device)
        out = torch.clip(action + noise, self.a_low, self.a_high)
        return out
    
    def soft_update(self, original):
        with torch.no_grad():
            for param1, param2 in zip(self.parameters(), original.parameters()):
                param1.copy_(param2 * self.tau + param1 * (1.0 - self.tau))
    def copy(self, original):
        with torch.no_grad():
            for param1, param2 in zip(self.parameters(), original.parameters()):
                param1.copy_(param2)

class QNet(nn.Module):
    def __init__(self, ob_dim, act_dim, tau):
        super().__init__()
        self.tau = tau
        self.l1 = nn.Linear(ob_dim + act_dim, 256)
        self.ac1 = nn.ReLU()
        self.l2 = nn.Linear(256, 256)
        self.ac2 = nn.ReLU()
        self.l3 = nn.Linear(256, 1)
    def forward(self, ob, act):
        inputs = torch.cat((ob, act), 1)
        out = self.ac1(self.l1(inputs))
        out = self.ac2(self.l2(out))
        out = self.l3(out)
        return out.squeeze()
    
    def soft_update(self, original):
        with torch.no_grad():
            for param1, param2 in zip(self.parameters(), original.parameters()):
                param1.copy_(param2 * self.tau + param1 * (1.0 - self.tau))
    def copy(self, original):
        with torch.no_grad():
            for param1, param2 in zip(self.parameters(), original.parameters()):
                param1.copy_(param2)

class ReplayBuffer:
    def __init__(self, config):
        self.max_len = config.buffer_size
        self.batch_size = config.batch_size
        self.memory = deque(maxlen=self.max_len) if self.max_len is not None else deque()
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

class DDPG(nn.Module):
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.seed = self.config.seed
        self.device = 'cpu'
        
        torch.manual_seed(self.seed)
        np.random.seed(seed=self.seed)
        random.seed(self.seed)
        
        self.env.reset(seed=self.seed)
        self.env.action_space.seed(self.seed)

        self.gamma = self.config.gamma
        self.ob_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

        self.actor = Actor(self.ob_dim, self.act_dim, self.config).to(self.device)
        self.actor_target = Actor(self.ob_dim, self.act_dim, self.config).to(self.device)
        self.actor_target.copy(self.actor)

        self.q = QNet(self.ob_dim, self.act_dim, self.config.tau).to(self.device)
        self.q_targ = QNet(self.ob_dim, self.act_dim, self.config.tau).to(self.device)
        self.q_targ.copy(self.q)

        self.buffer = ReplayBuffer(self.config)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.config.pi_lr)
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=self.config.v_lr)

        self.quant_input = torch.ao.quantization.QuantStub()
        self.dequant_output = torch.ao.quantization.DeQuantStub()

        self.num_iter = 0
    def to(self, device):
        model = super().to(device)
        model.device = device
        return model
    
    def compute_targets(self, next_states, rewards, done):
        next_states = torch.as_tensor(next_states, dtype=torch.float, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float, device=self.device)
        mu = self.actor_target(next_states) * self.config.a_high
        q_targs = self.q_targ(next_states, mu)
        targets = rewards + self.gamma * (1.0 - done) * q_targs
        return targets
    
    def update_q(self, states, actions, next_states, rewards, done):
        states = torch.as_tensor(states, dtype=torch.float, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float, device=self.device)
        targets = self.compute_targets(next_states, rewards, done)

        q = self.q(states, actions)
        loss = F.mse_loss(q, targets.detach())
        self.q_optim.zero_grad()
        loss.backward()
        self.q_optim.step()

    def update_actor(self, states):
        states = torch.as_tensor(states, dtype=torch.float, device=self.device)
        mu = self(states) * self.config.a_high
        loss = -torch.mean(self.q(states, mu))
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
    
    def forward(self, state):
        out = self.quant_input(state)
        out = self.actor(out)
        out = self.dequant_output(out)
        return out
    
    def train_agent(self):
        eval_x = []
        eval_y = []
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        state, _ = self.env.reset()
        done = False
        for t in range(self.config.max_timestamp):
            episode_timesteps += 1
            if t < self.config.start_steps:
                action = self.env.action_space.sample()
            else:
                action = self.actor.explore(state).detach().cpu().numpy()
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.buffer.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if t >= self.config.start_steps:
                self.train_iter()
            if done:
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                state, _ = self.env.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
            if (t + 1) % self.config.eval_freq == 0:
                eval_x.append(t)
                res = self.evaluation()
                eval_y.append(res)
                self.save_model(f"models/DDPG-{self.config.env_name}-seed-{self.seed}.pt")
        eval_x = np.array(eval_x)
        eval_y = np.array(eval_y)
        np.savez(f"results/{self.config.env_name}-seed-{self.seed}.npz", x=eval_x, y=eval_y)

    def train_iter(self):
        self.num_iter += 1
        states, actions, rewards, next_states, done = self.buffer.sample()
        self.update_q(states, actions, next_states, rewards, done)
        
        self.update_actor(states)
        self.q_targ.soft_update(self.q)
        self.actor_target.soft_update(self.actor)

    def evaluation(self, seed=None):
        env = gym.make(self.config.env)
        ep_reward = 0
        if seed is None:
            seed = self.config.seed
        state, _ = env.reset(seed = seed + 100)
        steps = 0
        for i in range(self.config.eval_epochs):
            state, _ = env.reset()
            done = False
            while not done:
                action = self(torch.as_tensor(state[None,:], dtype=torch.float, device=self.device)).detach().cpu().numpy().squeeze(axis=0)
                state, reward, terminated, truncated, _ = env.step(action*self.config.a_high)
                done = terminated or truncated
                ep_reward += reward
                steps += 1
            state, _ = env.reset()
            done = False
        print("---------------------------------------")
        print(f"Evaluation over {self.config.eval_epochs} episodes: {ep_reward/self.config.eval_epochs:.3f}")
        print("---------------------------------------")
        return ep_reward/self.config.eval_epochs, steps/self.config.eval_epochs
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    def load_model(self, path):
        self.load_state_dict(torch.load(path,map_location='cpu'))
