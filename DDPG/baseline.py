from config import *
import torch
import gymnasium as gym
from ddpg import DDPG
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
seed = [2]
fp32_time = []
fp32_step = []
fp32_return = []
fp32_ram = []
config = cfg(seed[0])
env = gym.make(config.env)
agent = DDPG(env, config).to('cpu')
agent.load_model(f"models/DDPG-{config.env_name}-seed-1.pt")
state, info = env.reset(seed=seed[0]+100)
with torch.no_grad():
    for i in range(10):
        origin_start = time.time()
        state, _ = env.reset()
        done = False
        r = 0.0
        step = 0
        while not done:
            action = agent(torch.as_tensor(state, dtype=torch.float)).detach().cpu().numpy()
            state, reward, terminated, truncated, _ = env.step(action)
            r += reward
            done = terminated or truncated
            step += 1
        origin_end = time.time()
        fp32_return.append(r)
        fp32_time.append(origin_end - origin_start)
        fp32_step.append(step)
        fp32_ram.append(psutil.Process().memory_info().rss / (1024 * 1024))
    print(f"{np.mean(fp32_return):.2f},{np.std(fp32_return):.2f},{np.mean(fp32_time):.2f},{np.std(fp32_time):.2f},{np.mean(fp32_step):.2f},{np.std(fp32_step):.2f},{np.mean(fp32_ram):.2f},{np.std(fp32_ram):.2f}")
