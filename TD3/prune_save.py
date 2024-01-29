from config import *
import torch
import gymnasium as gym
from td3 import TwinDelayedDDPG
import numpy as np
import time
import argparse
import psutil
import torch_pruning as tp 

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
agent = TwinDelayedDDPG(env,config)
agent.load_model(f"models/TD3-{config.env_name}-seed-1.pt")

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
    ignored_layers=[agent.actor.l3]
)
pruner.step()
agent.zero_grad()
torch.save(agent, f"models/pruning/TD3-{config.env_name}-{args.prune_amount}-l{args.n}.pth")
torch.onnx.export(agent, example_inputs, f"models/pruning/TD3-{config.env_name}-{args.prune_amount}-l{args.n}.onnx")