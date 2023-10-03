from config import *
import torch
import gymnasium as gym
from td3 import TwinDelayedDDPG
import numpy as np
import time

seed = [2,3,4,5,6,7,8,9,10,11]
fp32_time = []
fp32_step = []
fp32_return = []
config = HalfCheetahConfig(seed[0])
env = gym.make(config.env)
agent = TwinDelayedDDPG(env, config).to('cpu')
agent.load_model(f"models/TD3-{config.env_name}-seed-1.pt")
for i in range(len(seed)):
    origin_start = time.time()
    avg_return, steps_origin = agent.evaluation(seed=seed[i])
    origin_end = time.time()

    fp32_time.append(origin_end - origin_start)
    fp32_return.append(avg_return)
    fp32_step.append(steps_origin)
print(f"#### Task: {config.env_name}")
print()
print("|                     | fp32               |")
print("|---------------------|--------------------|")
print(f"| avg. return         | {np.mean(fp32_return):.2f} +/- {np.std(fp32_return):.2f}  |")
print(f"| avg. inference time |  {np.mean(fp32_time):.2f} +/- {np.std(fp32_time):.2f}     |")
print(f"| avg. ep length      | {np.mean(fp32_step):.2f} +/- {np.std(fp32_step):.2f}   |")
