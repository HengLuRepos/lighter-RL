from config import *
import torch
import gymnasium as gym
from td3 import TwinDelayedDDPG
import numpy as np
import time
import argparse
import psutil
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
def eval(agent, seed):
    return agent.evaluation(seed)

seed = [2,3,4,5,6,7,8,9,10,11]
fp32_time = []
fp32_step = []
fp32_return = []
fp32_ram = []
config = cfg(seed[0])
env = gym.make(config.env)
agent = TwinDelayedDDPG(env, config).to('cpu')
agent.load_model(f"models/TD3-{config.env_name}-seed-1.pt")


for i in range(len(seed)):
    origin_start = time.time()
    avg_return, steps_origin = eval(agent, seed[i])
    origin_end = time.time()
    fp32_ram.append(psutil.Process().memory_info().rss / (1024 * 1024))
    fp32_time.append(origin_end - origin_start)
    fp32_return.append(avg_return)
    fp32_step.append(steps_origin)

print(f"{np.mean(fp32_return):.2f},{np.std(fp32_return):.2f},{np.mean(fp32_time):.2f},{np.std(fp32_time):.2f},{np.mean(fp32_step):.2f},{np.std(fp32_step):.2f},{np.mean(fp32_ram):.2f},{np.std(fp32_ram):.2f}")


