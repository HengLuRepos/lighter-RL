from config import *
import torch
import gymnasium as gym
from ddpg_quantize import DDPG
import numpy as np
import time
from torch.ao.quantization.qconfig import QConfig, get_default_qat_qconfig
seed = [2,3,4,5,6,7,8,9,10,11]
int8_time = []
int8_step = []
int8_return = []
config = AntConfig(seed[0])
env = gym.make(config.env)
agent = DDPG(env, config).to('cpu')
agent.load_model(f"models/DDPG-{config.env_name}-seed-1.pt")

agent.eval()
agent.qconfig = get_default_qat_qconfig(backend='x86')
torch.backends.quantized.engine = 'x86'
torch.ao.quantization.quantize_dtype = torch.qint8
agent_prepared = torch.ao.quantization.prepare_qat(agent.train(), inplace=False)
agent_prepared.train()
agent_int8 = torch.ao.quantization.convert(agent_prepared.eval(), inplace=False)
agent_int8.load_model(f"models/qat/DDPG-{config.env_name}-default-x86.pt")
agent_int8.eval()
for i in range(len(seed)):
    quant_start = time.time()
    avg_return_int8, steps_quant = agent_int8.evaluation(seed[i])
    quant_end = time.time()
    int8_time.append(quant_end - quant_start)
    int8_return.append(avg_return_int8)
    int8_step.append(steps_quant)

print(f"#### Task: {config.env_name}")
print()
print("|                     | QAT               |")
print("|---------------------|--------------------|")
print(f"| avg. return         | {np.mean(int8_return):.2f} +/- {np.std(int8_return):.2f}  |")
print(f"| avg. inference time | {np.mean(int8_time):.2f} +/- {np.std(int8_time):.2f}      |")
print(f"| avg. ep length      | {np.mean(int8_step):.2f} +/- {np.std(int8_step):.2f}  |")

