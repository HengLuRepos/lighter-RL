from td3 import TwinDelayedDDPG
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
    config = AntConfig(seed[i])
    env = gym.make(config.env)
    td3 = TwinDelayedDDPG(env, config).to('cpu')
    td3.load_model(f"models/TD3-{config.env_name}-seed-1.pt")
    origin_start = time.time()
    avg_return, steps_origin = td3.evaluation()
    origin_end = time.time()


    td3_int8 = torch.quantization.quantize_dynamic(
        td3,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    quant_start = time.time()
    avg_return_int8, steps_quant = td3_int8.evaluation()
    quant_end = time.time()
    td3_int8.save_model(f"models/dynamic_quantize/td3-{config.env_name}.pt")

    fp32_time.append(origin_end - origin_start)
    int8_time.append(quant_end - quant_start)
    fp32_return.append(avg_return)
    int8_return.append(avg_return_int8)
    fp32_step.append(steps_origin)
    int8_step.append(steps_quant)
print(f"#### Task: {config.env_name}")
print()
print("|                     | fp32               | int8               |")
print("|---------------------|--------------------|--------------------|")
print(f"| avg. return         | {np.mean(fp32_return):.2f} +/- {np.std(fp32_return):.2f}  | {np.mean(int8_return):.2f} +/- {np.std(int8_return):.2f}  |")
print(f"| avg. inference time |  {np.mean(fp32_time):.2f} +/- {np.std(fp32_time):.2f}     | {np.mean(int8_time):.2f} +/- {np.std(int8_time):.2f}      |")
print(f"| avg. ep length      | {np.mean(fp32_step):.2f} +/- {np.std(fp32_step):.2f}   | {np.mean(int8_step):.2f} +/- {np.std(int8_step):.2f}  |")

