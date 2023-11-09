from config import *
import torch
import gymnasium as gym
from ddpg_quantize import DDPG
import numpy as np
import time
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
def fuse_modules(model):
    if hasattr(model, 'fuse_modules'):
        model.fuse_modules()
    for p in list(model.modules())[1:]:
        fuse_modules(p)
config = cfg(seed[0])
env = gym.make(config.env)
agent = DDPG(env, config).to('cpu')
agent.load_model(f"models/DDPG-{config.env_name}-seed-1.pt")
agent.eval()
agent.qconfig = torch.ao.quantization.get_default_qconfig('x86')
torch.backends.quantized.engine = 'x86'
torch.ao.quantization.quantize_dtype = torch.qint8
fuse_modules(agent)
agent_prepared = torch.ao.quantization.prepare(agent)
env_temp = gym.make(config.env)
state, _ = env_temp.reset(seed=seed[0] + 100)
for _ in range(1000):
    agent_prepared(torch.as_tensor(state[None,:], dtype=torch.float, device=agent_prepared.device))
    state = env_temp.observation_space.sample()
agent_int8 = torch.ao.quantization.convert(agent_prepared)
for i in range(len(seed)):  
    quant_start = time.time()
    avg_return_int8, steps_quant = agent_int8.evaluation(seed=seed[i])
    quant_end = time.time()
    agent_int8.save_model(f"models/static_quantize/DDPG-{config.env_name}.pt")
    int8_time.append(quant_end - quant_start)
    int8_return.append(avg_return_int8)
    int8_step.append(steps_quant)

print(f"#### Task: {config.env_name}")
print()
print("|                 | int8-ptsq               |")
print("|--------------------|--------------------|")
print(f"| avg. return         | {np.mean(int8_return):.2f} +/- {np.std(int8_return):.2f}  |")
print(f"| avg. inference time | {np.mean(int8_time):.2f} +/- {np.std(int8_time):.2f}      |")
print(f"| avg. ep length      | {np.mean(int8_step):.2f} +/- {np.std(int8_step):.2f}  |")

