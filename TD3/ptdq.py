from td3_quantize import TwinDelayedDDPG
from config import *
import gymnasium as gym
import torch
import time
import numpy as np
seed = [2,3,4,5,6,7,8,9,10,11]
int8_time = []
int8_step = []
int8_return = []

config = AntConfig(seed[0])
env = gym.make(config.env)
td3 = TwinDelayedDDPG(env, config).to('cpu')
td3.load_model(f"models/TD3-{config.env_name}-seed-1.pt")
td3_int8 = torch.quantization.quantize_dynamic(
    td3,
    {torch.nn.Linear, torch.nn.ReLU},
    dtype=torch.qint8
)
for i in range(len(seed)):
    quant_start = time.time()
    avg_return_int8, steps_quant = td3_int8.evaluation(seed=seed[i])
    quant_end = time.time()
    td3_int8.save_model(f"models/dynamic_quantize/td3-{config.env_name}.pt")

    int8_time.append(quant_end - quant_start)
    int8_return.append(avg_return_int8)
    int8_step.append(steps_quant)
print(f"#### Task: {config.env_name}")
print()
print("|                     | int8-ptdq               |")
print("|---------------------|--------------------|")
print(f"| avg. return         | {np.mean(int8_return):.2f} +/- {np.std(int8_return):.2f}  |")
print(f"| avg. inference time | {np.mean(int8_time):.2f} +/- {np.std(int8_time):.2f}      |")
print(f"| avg. ep length      | {np.mean(int8_step):.2f} +/- {np.std(int8_step):.2f}  |")

