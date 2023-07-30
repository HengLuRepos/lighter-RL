import torch
import torch.nn as nn
import torch.distributions as ptd
import numpy as np
import gymnasium as gym
import torch.nn.functional as F
from collections import deque
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def np2torch(np_arr):
    np_arr = torch.from_numpy(np_arr) if isinstance(np_arr,np.ndarray) else np_arr
    return np_arr.to(device).float()

class Actor(nn.Module):
    def __init__(self, ob_dim, act_dim, config):
        super().__init__()
        self.config = config
        self.a_low = self.config.a_low
        self.a_high = self.config.a_high
        self.explore_noise = self.config.explore_noise
        self.target_noise = self.config.target_noise
        self.noise_clip = self.config.noise_clip
        self.tau = self.config.tau

        self.l1 = nn.Linear(ob_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, act_dim)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.l1.weight)
        nn.init.xavier_normal_(self.l2.weight)
        nn.init.xavier_normal_(self.l3.weight)

    def forward(self, ob):
        out = F.relu(self.l1(ob))
        out = F.relu(self.l2(out))
        out = F.tanh(self.l3(out)) * self.a_high
        return out
    
    def explore(self, state):
        state = np2torch(state)
        action = self(state)
        noise = torch.normal(mean=0.0, std=self.explore_noise, size=action.size(), device=device)
        out = torch.clip(action + noise, self.a_low, self.a_high)
        return out

    def target(self, next_actions):
        noise = torch.normal(mean=0.0, std=self.target_noise, size=next_actions.size(), device=device)
        actions = torch.clip(next_actions + torch.clip(noise, -self.noise_clip, self.noise_clip), self.a_low, self.a_high)
        return actions
    
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
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
    def forward(self, ob, act):
        inputs = torch.cat((ob, act), 1)
        out = F.relu(self.l1(inputs))
        out = F.relu(self.l2(out))
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

class TwinDelayedDDPG(nn.Module):
    def __init__(self, env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.seed = self.config.seed
        
        torch.manual_seed(self.seed)
        np.random.seed(seed=self.seed)
        self.env.reset(seed=self.seed)
        self.env.action_space.seed(self.seed)

        self.gamma = self.config.gamma
        self.ob_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

        self.actor = Actor(self.ob_dim, self.act_dim, self.config).to(device)
        self.actor_target = Actor(self.ob_dim, self.act_dim, self.config).to(device)
        self.actor_target.copy(self.actor)

        self.q1 = QNet(self.ob_dim, self.act_dim, self.config.tau).to(device)
        self.q1_targ = QNet(self.ob_dim, self.act_dim, self.config.tau).to(device)
        self.q1_targ.copy(self.q1)

        self.q2 = QNet(self.ob_dim, self.act_dim, self.config.tau).to(device)
        self.q2_targ = QNet(self.ob_dim, self.act_dim, self.config.tau).to(device)
        self.q2_targ.copy(self.q2)

        self.buffer = ReplayBuffer(self.config)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.config.pi_lr)
        self.q1_optim = torch.optim.Adam(self.q1.parameters(), lr=self.config.v_lr)
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=self.config.v_lr)
    
    def compute_targets(self, next_states, rewards, done):
        next_states = np2torch(next_states)
        rewards = np2torch(rewards)
        done = np2torch(done)
        mu = self.actor_target(next_states)
        actions = self.actor.target(mu)
        q_targs = torch.min(self.q1_targ(next_states, actions), self.q2_targ(next_states, actions))
        targets = rewards + self.gamma * (1.0 - done) * q_targs
        return targets
    
    def update_q(self, states, actions, next_states, rewards, done):
        states = np2torch(states)
        actions = np2torch(actions)
        targets = self.compute_targets(next_states, rewards, done)

        q1 = self.q1(states, actions)
        loss1 = F.mse_loss(q1, targets.detach())
        self.q1_optim.zero_grad()
        loss1.backward()
        self.q1_optim.step()

        q2 = self.q2(states, actions)
        loss2 = F.mse_loss(q2, targets.detach())
        self.q2_optim.zero_grad()
        loss2.backward()
        self.q2_optim.step()
    
    def update_actor(self, states):
        states = np2torch(states)
        mu = self.actor(states)
        loss = -torch.mean(self.q1(states, mu))
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
    
    def initial_explore(self):
        state, _ = self.env.reset()
        for step in range(self.config.start_steps):
            action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.buffer.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                state, _ = self.env.reset()
    
    def train_agent(self):
        self.initial_explore()
        ep_rewards = []
        best_return = None
        for ep in range(self.config.epoch):
            state, _ = self.env.reset()
            for step in range(self.config.steps_per_epoch):
                action = self.actor.explore(state).detach().cpu().numpy()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.buffer.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    state, _ = self.env.reset()

            for j in range(self.config.update_freq):
                states, actions, rewards, next_states, done = self.buffer.sample()
                self.update_q(states, actions, next_states, rewards, done)
                if j % self.config.policy_delay == 0:
                    self.update_actor(states)
                    self.q1_targ.soft_update(self.q1)
                    self.q2_targ.soft_update(self.q2)
                    self.actor_target.soft_update(self.actor)

            state, _ = self.env.reset()
            done = False
            ep_reward = 0
            while not done:
                action = self.actor(np2torch(state)).detach().cpu().numpy()
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                ep_reward += reward
            if best_return is None or ep_reward >= max(ep_rewards):
                best_return = ep_reward
                self.save_model("models/TD3-{}-seed-{}-best.pt".format(self.config.env_name, self.seed))
            self.save_model("models/TD3-{}-seed-{}.pt".format(self.config.env_name, self.seed))
            ep_rewards.append(ep_reward)
            print("Epoch {}: Episodic return: {:.2f}".format(ep, ep_reward))
        print("Maximun Ep return: {:.2f}".format(max(ep_rewards)))
    
    def evaluation(self):
        results = []
        for ep in range(self.config.eval_epochs):
            ep_return = 0
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.actor(np2torch(state)).detach().cpu().numpy()
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                ep_return = self.gamma * ep_return + reward
            results.append(ep_return)
        print("Avg Episodic Return: {:.2f}".format(np.mean(results)))
        return results

    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
