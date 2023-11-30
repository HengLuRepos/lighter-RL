from config import *
import torch
import gymnasium as gym
from trpo import TRPO
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
seed = [2,3,4,5,6,7,8,9,10,11]
fp32_time = []
fp32_step = []
fp32_return = []
fp32_ram = []
config = cfg(seed[0])
env = gym.make(config.env)
agent = TRPO(env, config).to('cpu')
agent.load_model(f"models/trpo-{config.env_name}-seed-1.pt")