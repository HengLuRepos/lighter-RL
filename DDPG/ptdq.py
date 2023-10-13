from ddpg_quantize import DDPG
from config import *
import gymnasium as gym
import torch
import time
import numpy as np
seed = [2,3,4,5,6,7,8,9,10,11]
fp32_time = []
int8_time = []
fp32_step = []
int8_step = []
fp32_return = []
int8_return = []

for i in range(len(seed)):
    config = HumanoidStandupConfig(seed[i])
    env = gym.make(config.env)
    td3 = DDPG(env, config)
    td3.load_model(f"models/DDPG-{config.env_name}-seed-1.pt")


    td3_int8 = torch.quantization.quantize_dynamic(
        td3,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    quant_start = time.time()
    avg_return_int8, steps_quant = td3_int8.evaluation()
    quant_end = time.time()
    td3_int8.save_model(f"models/dynamic_quantize/DDPG-{config.env_name}.pt")

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
