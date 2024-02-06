from ddpg_quantize import DDPG
from config import *
import gymnasium as gym
import torch
import time
import numpy as np
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
int8_time = []
fp32_step = []
int8_step = []
int8_ram = []
fp32_return = []
int8_return = []
config = cfg(seed[1])
env = gym.make(config.env)
td3 = DDPG(env, config)
td3.load_model(f"models/DDPG-{config.env_name}-seed-1.pt")
td3_int8 = torch.quantization.quantize_dynamic(
    td3,
    {torch.nn.Linear},
    dtype=torch.qint8
)
td3_int8.save_model(f"models/dynamic_quantize/DDPG-{config.env_name}.pt")
state, info = env.reset(seed=seed[0]+100)
for i in range(10):
    quant_start = time.time()
    state, _ = env.reset()
    done = False
    r = 0.0
    step = 0
    while not done:
        action = td3_int8(torch.as_tensor(state, dtype=torch.float)).detach().cpu().numpy()
        state, reward, terminated, truncated, _ = env.step(action)
        r += reward
        done = terminated or truncated
        step += 1
    
    quant_end = time.time()

    int8_time.append(quant_end - quant_start)
    int8_return.append(r)
    int8_step.append(step)
    int8_ram.append(psutil.Process().memory_info().rss / (1024 * 1024))

print(f"{np.mean(int8_return):.2f},{np.std(int8_return):.2f},{np.mean(int8_time):.2f},{np.std(int8_time):.2f},{np.mean(int8_step):.2f},{np.std(int8_step):.2f},{np.mean(int8_ram):.2f},{np.std(int8_ram):.2f}")
