from config import *
import torch
import gymnasium as gym
from ddpg_quantize import DDPG
import numpy as np
import time
from torch.ao.quantization.qconfig import QConfig, get_default_qat_qconfig
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
int8_time = []
int8_step = []
int8_return = []
int8_ram = []
config = cfg(seed[0])
env = gym.make(config.env)
agent = DDPG(env, config).to('cpu')
#agent.load_model(f"models/DDPG-{config.env_name}-seed-1.pt")
agent.eval()
agent.qconfig = get_default_qat_qconfig(backend='x86')
torch.backends.quantized.engine = 'x86'
torch.ao.quantization.quantize_dtype = torch.qint8
agent_prepared = torch.ao.quantization.prepare_qat(agent.train(), inplace=False)
agent_prepared.train()
agent_int8 = torch.ao.quantization.convert(agent_prepared.eval(), inplace=False)
del agent, agent_prepared
agent_int8.load_model(f"models/qat/DDPG-{config.env_name}-default-x86.pt")
agent_int8.eval()
env.reset(seed=seed[0]+100)
for i in range(10):
    quant_start = time.time()
    state, _ = env.reset()
    done = False
    r = 0.0
    step = 0
    while not done:
        action = agent_int8(torch.as_tensor(state[None,:], dtype=torch.float)).detach().cpu().numpy().squeeze(axis=0)
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

