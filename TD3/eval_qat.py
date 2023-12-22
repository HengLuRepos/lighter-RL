from config import *
import torch
import gymnasium as gym
from td3_quantize import TwinDelayedDDPG
import numpy as np
import time
from torch.ao.quantization.qconfig import QConfig, get_default_qat_qconfig
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
int8_time = []
int8_step = []
int8_return = []
fp32_ram = []
config = cfg(seed[0])
env = gym.make(config.env)
agent = TwinDelayedDDPG(env, config).to('cpu')
agent.load_model(f"models/TD3-{config.env_name}-seed-1.pt")
    
agent.eval()
agent.qconfig = get_default_qat_qconfig(backend='x86')
torch.backends.quantized.engine = 'x86'
torch.ao.quantization.quantize_dtype = torch.qint8
agent_prepared = torch.ao.quantization.prepare_qat(agent.train(), inplace=False)
agent_prepared.train()
agent_int8 = torch.ao.quantization.convert(agent_prepared.eval(), inplace=False)
agent_int8.load_model(f"models/qat/TD3-{config.env_name}-default-x86.pt")
agent_int8.eval()
for i in range(len(seed)):
    quant_start = time.time()
    avg_return_int8, steps_quant = agent_int8.evaluation(seed[i])
    quant_end = time.time()
    fp32_ram.append(psutil.Process().memory_info().rss / (1024 * 1024))
    int8_time.append(quant_end - quant_start)
    int8_return.append(avg_return_int8)
    int8_step.append(steps_quant)
    
print(f"{np.mean(int8_return):.2f},{np.std(int8_return):.2f},{np.mean(int8_time):.2f},{np.std(int8_time):.2f},{np.mean(int8_step):.2f},{np.std(int8_step):.2f},{np.mean(fp32_ram):.2f},{np.std(fp32_ram):.2f}")

