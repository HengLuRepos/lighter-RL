from config import *
import torch
import gymnasium as gym
from trpo import TRPO
import numpy as np
import time
import argparse
import psutil
import torch.nn.utils.prune as tp 
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
    parser.add_argument("--dim", type=int, default=1)
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

for name, module in agent.actor.named_modules():
    # prune 20% of connections in all 2D-conv layers
    if isinstance(module, torch.nn.Linear):
        tp.l1_unstructured(module, name='weight', amount=args.prune_amount)
        #tp.ln_structured(module, name='weight', amount=args.prune_amount, dim=args.dim, n=args.n)
for seed in eval_seed:
    steps = 0
    returns = 0
    start_time = time.time()
    returns, steps = agent.evaluation(seed=seed)
    end_time = time.time()
    fp32_ram.append(psutil.Process().memory_info().rss / (1024 * 1024))
    fp32_time.append(end_time- start_time)
    fp32_return.append(returns)
    fp32_step.append(steps)
print(f"{np.mean(fp32_return):.2f},{np.std(fp32_return):.2f},{np.mean(fp32_time):.2f},{np.std(fp32_time):.2f},{np.mean(fp32_step):.2f},{np.std(fp32_step):.2f},{np.mean(fp32_ram):.2f},{np.std(fp32_ram):.2f}")
