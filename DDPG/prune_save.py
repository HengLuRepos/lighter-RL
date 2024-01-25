from config import *
import torch
import gymnasium as gym
from ddpg import DDPG
import numpy as np
import time
import argparse
import psutil
import torch_pruning as tp
import torch.onnx
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
    parser.add_argument("--n",type=int,default=1)
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
agent = DDPG(env,config)
agent.load_model(f"models/DDPG-{config.env_name}-seed-1.pt")

agent.eval()

example_inputs = torch.as_tensor(env.observation_space.sample()[np.newaxis,:], dtype=torch.float)
imp = tp.importance.MagnitudeImportance(p=args.n)

pruner = tp.pruner.MagnitudePruner(
    agent,
    example_inputs,
    imp,
    pruning_ratio=args.prune_amount,
    ignored_layers=[agent.actor.l3]
)
pruner.step()
agent.zero_grad()
print(agent)
torch.save(agent, f"models/pruning/DDPG-{config.env_name}-{args.prune_amount}-l{args.n}.pth")
torch.onnx.export(agent, example_inputs, f"models/pruning/DDPG-{config.env_name}-{args.prune_amount}-l{args.n}.onnx")