from config import *
import torch
import gymnasium as gym
from trpo import TRPO
import numpy as np
import time
seed = [2,3,4,5,6,7,8,9,10,11]
#fp32_time = []
int8_time = []
#fp32_step = []
int8_step = []
#fp32_return = []
int8_return = []
def fuse_modules(model):
    if hasattr(model, 'fuse_modules'):
        model.fuse_modules()
    for p in list(model.modules())[1:]:
        fuse_modules(p)
for i in range(len(seed)):
    config = HalfCheetahConfig(seed[i])
    env = gym.make(config.env)
    agent = TRPO(env, config).to('cpu')
    agent.load_model(f"models/trpo-{config.env_name}-seed-1.pt")
    #origin_start = time.time()
    #avg_return, steps_origin = agent.evaluation(seed=seed[i])
    #origin_end = time.time()

    agent.eval()
    agent.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    torch.backends.quantized.engine = 'x86'
    torch.ao.quantization.quantize_dtype = torch.qint8
    fuse_modules(agent)
    agent_prepared = torch.ao.quantization.prepare(agent)
    env_temp = gym.make(config.env)
    state, _ = env_temp.reset(seed=seed[i] + 100)
    for _ in range(1000):
        agent_prepared(torch.as_tensor(state[None,:], dtype=torch.float, device=agent_prepared.device))
        state = env_temp.observation_space.sample()
    agent_int8 = torch.ao.quantization.convert(agent_prepared)
    
    quant_start = time.time()
    avg_return_int8, steps_quant = agent_int8.evaluation(seed=seed[i])
    quant_end = time.time()
    agent_int8.save_model(f"models/static_quantize/trpo-{config.env_name}.pt")
    #fp32_time.append(origin_end - origin_start)
    int8_time.append(quant_end - quant_start)
    #fp32_return.append(avg_return)
    int8_return.append(avg_return_int8)
    #fp32_step.append(steps_origin)
    int8_step.append(steps_quant)

print(f"#### Task: {config.env_name}")
print()
print("|                 | int8-ptsq               |")
print("|--------------------|--------------------|")
print(f"| avg. return         | {np.mean(int8_return):.2f} +/- {np.std(int8_return):.2f}  |")
print(f"| avg. inference time | {np.mean(int8_time):.2f} +/- {np.std(int8_time):.2f}      |")
print(f"| avg. ep length      | {np.mean(int8_step):.2f} +/- {np.std(int8_step):.2f}  |")

