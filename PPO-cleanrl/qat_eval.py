# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.ao.quantization.qconfig import QConfig, get_default_qat_qconfig


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--layer-size", type=int, default=64,
        help="hidden layer size")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, layer_size):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), layer_size)),
            nn.Tanh(),
            layer_init(nn.Linear(layer_size, layer_size)),
            nn.Tanh(),
            layer_init(nn.Linear(layer_size, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), layer_size)),
            nn.Tanh(),
            layer_init(nn.Linear(layer_size, layer_size)),
            nn.Tanh(),
            layer_init(nn.Linear(layer_size, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        self.quant_state = torch.ao.quantization.QuantStub()
        self.dequant_mean = torch.ao.quantization.DeQuantStub()
        self.dequant_std = torch.ao.quantization.DeQuantStub()

    def get_value(self, x):
        return self.critic(x)
    def forward(self, x):
        x = self.quant_state(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        action_mean = self.dequant_mean(action_mean)
        action_std = self.dequant_std(action_std)
        return action_mean, action_std
    def get_action_and_value(self, x, action=None):
        action_mean, action_std = self(x)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
    



if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = 'cpu'

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    fp32_time = []
    fp32_step = []
    fp32_return = []

    agent = Agent(envs, args.layer_size).to(device)
    agent.eval()
    agent.qconfig = get_default_qat_qconfig(backend='x86')
    torch.backends.quantized.engine = 'x86'
    torch.ao.quantization.quantize_dtype = torch.qint8
    agent_prepared = torch.ao.quantization.prepare_qat(agent.train(), inplace=False)
    agent_prepared.train()
    agent_int8 = torch.ao.quantization.convert(agent_prepared.eval(), inplace=False)
    agent_int8.load_model(f"models/qat/ppo-{args.env_id}-seed-{args.seed}.pt")
    agent_int8.eval()    
    seeds = [2,3,4,5,6,7,8,9,10,11]
    for seed in seeds:
      steps = 0
      returns = 0
      start_time = time.time()
      states, _ = envs.reset(seed=seed + 100)
      for i in range(args.update_epochs):
        done = False
        while not done:
          action,_ = agent_int8(torch.as_tensor(states, dtype=torch.float32))
          states, reward, ter, trun, _ = envs.step(action.detach().numpy())
          steps += 1
          done = any(ter or trun)
          returns += reward
      end_time = time.time()
      fp32_time.append(end_time- start_time)
      fp32_return.append(returns/args.update_epochs)
      fp32_step.append(steps/args.update_epochs)
    print(f"#### Task: {args.env_id}")
    print()
    print("|                     | int8               |")
    print("|---------------------|--------------------|")
    print(f"| avg. return         | {np.mean(fp32_return):.2f} +/- {np.std(fp32_return):.2f}  |")
    print(f"| avg. inference time |  {np.mean(fp32_time):.2f} +/- {np.std(fp32_time):.2f}     |")
    print(f"| avg. ep length      | {np.mean(fp32_step):.2f} +/- {np.std(fp32_step):.2f}   |")
    envs.close()
