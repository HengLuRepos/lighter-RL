from config import *
import torch
import gymnasium as gym
from trpo import TRPO
import numpy as np
import time
from torch.ao.quantization.qconfig import QConfig, get_default_qat_qconfig
from torch.ao.quantization.fake_quantize import default_fused_act_fake_quant,default_fused_wt_fake_quant
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, default_per_channel_weight_observer
seed = [2,3,4,5,6,7,8,9,10,11]
fp32_time = []
int8_time = []
fp32_step = []
int8_step = []
fp32_return = []
int8_return = []
for i in range(len(seed)):
    config = HalfCheetahConfig(seed[i])
    env = gym.make(config.env)
    agent = TRPO(env, config).to('cpu')
    agent.load_model(f"models/trpo-{config.env_name}-seed-1.pt")
    origin_start = time.time()
    agent.eval()
    #avg_return, steps_origin = evaluation(agent.to('cpu'), config)
    avg_return, steps_origin = agent.evaluation(seed[i])
    origin_end = time.time()

    agent.eval()
    agent.qconfig = get_default_qat_qconfig(backend='qnnpack')
    torch.backends.quantized.engine = 'x86'
    torch.ao.quantization.quantize_dtype = torch.qint8
    agent_prepared = torch.ao.quantization.prepare_qat(agent.train(), inplace=False)
    agent_prepared.train()
    agent_int8 = torch.ao.quantization.convert(agent_prepared.eval(), inplace=False)
    agent_int8.load_model(f"models/qat/trpo-{config.env_name}-default-qnnpack.pt")
    agent_int8.eval()
    quant_start = time.time()
    avg_return_int8, steps_quant = agent_int8.evaluation(seed[i])
    quant_end = time.time()
    #agent_int8.save_model(f"models/qat/trpo-{config.env_name}.pt")
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

