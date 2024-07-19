from config import *
import torch
import gymnasium as gym
from trpo import TRPO
import numpy as np
import time
import argparse
import psutil
import torch_pruning as tp 
from trpo import TRPO

env_map = {
    "HalfCheetah-v4": HalfCheetahConfig,
    "Humanoid-v4": HumanoidConfig,
    "HumanoidStandup-v4": HumanoidStandupConfig,
    "Ant-v4": AntConfig,
    "Hopper-v4": HopperConfig,

}
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4",
        help="the id of the environment")
    parser.add_argument("--prune-amount", type=float, default=0.1,
        help="the id of the environment")
    parser.add_argument("--n",type=int,default=2)
    args = parser.parse_args()
    
    return args
args = parse_args()
cfg = env_map[args.env_id]
config = cfg(1)
env = gym.make(config.env)
fp32_time = []
fp32_step = []
fp32_return = []
fp32_ram = []
eval_seed = [2,3,4,5,6,7,8,9,10,11]
agent = TRPO(env,config)
agent.load_model(f"models/trpo-{config.env_name}-seed-1.pt")

agent.eval()

example_inputs = torch.as_tensor(env.observation_space.sample()[np.newaxis,:], dtype=torch.float)
if args.n == 1:
    imp = tp.importance.MagnitudeImportance(p=args.n, normalizer=None, group_reduction="first")
else:
    imp = tp.importance.MagnitudeImportance(p=args.n, normalizer='max', group_reduction="first")
pruner = tp.pruner.MagnitudePruner(
    agent,
    example_inputs,
    imp,
    pruning_ratio=args.prune_amount,
    ignored_layers=[agent.actor.out]
)
pruner.step()
agent.zero_grad()
torch.save(agent, f"models/pruning/TRPO-{config.env_name}-{args.prune_amount}-l{args.n}.pth")
