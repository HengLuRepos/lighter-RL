from config import *
import torch
import gymnasium as gym
from trpo import TRPO
import numpy as np
import time
import psutil
import argparse
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
fp32_return = []
int8_return = []
fp32_ram = []
config = cfg(seed[0])
env = gym.make(config.env)
agent = TRPO(env, config).to('cpu')
agent.load_model(f"models/trpo-{config.env_name}-seed-1.pt")
agent_int8 = torch.ao.quantization.quantize_dynamic(
    agent,
    {torch.nn.Linear, torch.nn.ReLU},
    dtype=torch.qint8
)
agent_int8.save_model(f"models/dynamic_quantize/TRPO-{config.env_name}.pt")
for i in range(len(seed)):
    
    quant_start = time.time()
    avg_return_int8, steps_quant = agent_int8.evaluation(seed=seed[i])
    quant_end = time.time()
    
    fp32_ram.append(psutil.Process().memory_info().rss / (1024 * 1024))
    #fp32_time.append(origin_end - origin_start)
    int8_time.append(quant_end - quant_start)
    #fp32_return.append(avg_return)
    int8_return.append(avg_return_int8)
    #fp32_step.append(steps_origin)
    int8_step.append(steps_quant)

print(f"{np.mean(int8_return):.2f},{np.std(int8_return):.2f},{np.mean(int8_time):.2f},{np.std(int8_time):.2f},{np.mean(int8_step):.2f},{np.std(int8_step):.2f},{np.mean(fp32_ram):.2f},{np.std(fp32_ram):.2f}")

