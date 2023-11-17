from config import *
import torch
import gymnasium as gym
from td3 import TwinDelayedDDPG
import numpy as np
import time
import argparse
import psutil
import torch.nn.utils.prune as tp 

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
    args = parser.parse_args()
    
    return args
args = parse_args()
cfg = env_map[args.env_id]
config = cfg(1)
env = gym.make(config.env)
eval_seed = 100
agent = TwinDelayedDDPG(env,config)
agent.load_model(f"models/TD3-{config.env_name}-seed-1.pt")
time1 = time.time()
avg_return1, steps_origin1 = agent.evaluation(4)
time1_1 = time.time()
for name, module in agent.actor.named_modules():
    # prune 20% of connections in all 2D-conv layers
    if isinstance(module, torch.nn.Linear):
        tp.l1_unstructured(module, name='weight', amount=0.4)
time2 = time.time()
avg_return2, steps_origin2 = agent.evaluation(eval_seed)
time2_2 = time.time()
print(f"Before pruning: return {avg_return1:.2f}, steps {steps_origin1:.2f}, time {time1_1-time1}")

print(f"After pruning: return {avg_return2:.2f}, steps {steps_origin2:.2f}, time {time2_2-time2}")