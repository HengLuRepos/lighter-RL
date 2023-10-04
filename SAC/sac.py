import gymnasium as gym
import torch
import torch.nn as nn
import torch.distributions as ptd
import numpy as np
from collections import deque
import random
from stable_baselines3.common.buffers import ReplayBuffer
#part from cleanRL
class MLP(nn.Module):
    def __init__(self, input_dim, layer_size):
        super().__init__()
        self.l1 = nn.Linear(input_dim, layer_size)
        self.ac1 = nn.ReLU()
        self.l2 = nn.Linear(layer_size, layer_size)
        self.ac2 = nn.ReLU()

        self.init_weights()
    def forward(self, x):
        out = self.ac1(self.l1(x))
        out = self.ac2(self.l2(out))
        return out
    def init_weights(self):
        nn.init.xavier_normal_(self.l1.weight)
        nn.init.xavier_normal_(self.l2.weight)

class Actor(nn.Module):
    def __init__(self, env:gym.Env, config):
        super().__init__()
        self.network = MLP(np.prod(env.observation_space.shape),
                           config.layer_size)
        self.mean_out = nn.Linear(config.layer_size, np.prod(env.action_space.shape))
        self.logstd_out = nn.Linear(config.layer_size, np.prod(env.action_space.shape))
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5
        
        self.action_scale = nn.Parameter(torch.FloatTensor((env.action_space.high -
                                               env.action_space.low) / 2.), 
                                               requires_grad=False)
        self.action_bias = nn.Parameter(torch.FloatTensor((env.action_space.high +
                                              env.action_space.low) / 2.), 
                                              requires_grad=False)
        
    def forward(self, x):
        x = self.network(x)
        mean = self.mean_out(x)
        log_std = nn.functional.tanh(self.logstd_out(x))
        #log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def act(self, x):
        mean, logstd = self(x)
        std = logstd.exp()
        dist = ptd.Normal(mean, std)
        action_sampled = dist.rsample()
        scaled = torch.tanh(action_sampled)
        action = scaled * self.action_scale + self.action_bias
        log_probs = dist.log_prob(action_sampled)
        log_probs -= torch.log(self.action_scale * (1-scaled.pow(2))+1e-6)
        log_probs = log_probs.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_probs, mean
    
class SoftQNet(nn.Module):
    def __init__(self, env:gym.Env, config):
        super().__init__()
        self.network = MLP(np.prod(env.observation_space.shape) + 
                           np.prod(env.action_space.shape),
                           config.layer_size)
        self.q_out = nn.Linear(config.layer_size, 1)
    
    def forward(self, state, action):
        x = torch.cat((state, action), 1)
        x = self.network(x)
        x = self.q_out(x)
        return x

class SAC(nn.Module):
    def __init__(self, env:gym.Env, config):
        super().__init__()
        self.env = env
        self.config = config
        self.seed = self.config.seed

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.q_lr = self.config.q_lr
        self.pi_lr = self.config.pi_lr
        self.alpha_lr = self.config.alpha_lr
        self.gamma = self.config.gamma
        self.tau = self.config.tau

        self.actor = Actor(self.env, self.config)
        self.q1 = SoftQNet(self.env, self.config)
        self.q2 = SoftQNet(self.env, self.config)
        self.q1_target = SoftQNet(self.env, self.config)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target = SoftQNet(self.env, self.config)
        self.q2_target.load_state_dict(self.q2.state_dict())
        if self.config.alpha_tune:
            self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
        else:
            self.alpha = nn.Parameter(torch.tensor(self.config.alpha, requires_grad=False))

    def forward(self, state):
        out = self.actor(state)[0]
        return out
    
    def train_agent(self):
        episide = 0
        episode_reward = 0
        best_eval = None
        envs = gym.vector.make(self.config.env, num_envs=1, asynchronous=False)
        device = None
        q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.q_lr)
        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.pi_lr)
        if self.config.alpha_tune:
            target_entropy = -torch.prod(torch.tensor(envs.single_action_space.shape)).item()
            alpha = self.log_alpha.exp().item()
            alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
            device = self.log_alpha.device
        else:
            alpha = self.alpha
            device = self.alpha.device
        
        envs.single_observation_space.dtype = np.float32
        buffer = ReplayBuffer(
            self.config.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device=device,
            handle_timeout_termination=True,
        )
        state, info = envs.reset(seed = self.seed)
        for steps in range(self.config.max_timesteps):
            if steps < self.config.start_steps:
                action = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                action, _, _ = self.actor.act(torch.as_tensor(state, dtype=torch.float, device=device))
                action = action.detach().cpu().numpy()
            next_state, reward, terminated, truncated, infos = envs.step(action)
            done = terminated or truncated
            infos = [infos]
            for info in infos:
                if "episode" in info.keys():
                    print(f"global_step={steps}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], steps)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], steps)
                    break

            real_next_obs = next_state.copy()
            for idx, d in enumerate(done):
                if d:
                    real_next_obs[idx] = infos[idx]["final_observation"][0]

            buffer.add(state, real_next_obs, action, reward, done, infos)
            state = next_state
            episode_reward += reward.sum()/envs.num_envs
            if any(done):
                print(f"{self.config.env_name} Episode {episide} total rewards: {episode_reward:04.2f}")
                episide += 1
                episode_reward = 0
            if steps >= self.config.start_steps:
                batch = buffer.sample(self.config.batch_size)
                with torch.no_grad():
                    next_actions, next_logp, _ = self.actor.act(batch.next_observations)
                    q1_target = self.q1_target(batch.next_observations, next_actions)
                    q2_target = self.q1_target(batch.next_observations, next_actions)
                    q_target = torch.min(q1_target, q2_target) - alpha * next_logp
                    qs = batch.rewards.flatten() + (1 - batch.dones.flatten()) * self.gamma * (q_target).view(-1)
                q1 = self.q1(batch.observations, batch.actions).view(-1)
                q2 = self.q2(batch.observations, batch.actions).view(-1)
                q_loss = nn.functional.mse_loss(q1, qs) + nn.functional.mse_loss(q2, qs)
                q_optimizer.zero_grad()
                q_loss.backward()
                q_optimizer.step()

                if steps % self.config.policy_freq == 0:
                    for _ in range(self.config.policy_freq):
                        next_actions, next_logp, _ = self.actor.act(batch.observations)
                        q1 = self.q1(batch.observations, next_actions)
                        q2 = self.q2(batch.observations, next_actions)
                        q = torch.min(q1, q2).view(-1)
                        actor_loss = ((alpha * next_logp) - q).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                        if self.config.alpha_tune:
                            with torch.no_grad():
                                _, logp, _ = self.actor.act(batch.observations)
                            alpha_loss = (-self.log_alpha.exp() * (logp + target_entropy)).mean()
                            alpha_optimizer.zero_grad()
                            alpha_loss.backward()
                            alpha_optimizer.step()
                            alpha = self.log_alpha.exp().item()

                for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                self.save_model(f"models/sac-{self.config.env_name}-seed-{self.seed}.pt")


                """ if steps % self.config.eval_freq == 0:
                    avg_return, avg_step = self.evaluation()
                    print(f"{self.config.env_name} Evaluation: avg return: {avg_return:04.2f} avg length: {avg_step:04.2f}")
                    if best_eval is None or best_eval < avg_return:
                        best_eval = avg_return
                        self.save_model(f"models/sac-{self.config.env_name}-seed-{self.seed}-best.pt") """
            envs.close()

    def save_model(self, path):
        torch.save(self.state_dict(), path)
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
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
                action = self.actor.act(torch.as_tensor(state, dtype=torch.float, device=self.actor.mean_out.weight.device))[2]
                action = action.detach().cpu().numpy()
                next_state, reward, terminated, truncated, _ = env.step(action)
                returns += reward
                state = next_state
                done = terminated or truncated
                steps += 1
            state, _ = env.reset()
        return returns/self.config.eval_epochs, steps/self.config.eval_epochs

              