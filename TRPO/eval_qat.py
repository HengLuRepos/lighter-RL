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
int8_time = []
int8_step = []
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

    agent.eval()
    agent.qconfig = get_default_qat_qconfig(backend='qnnpack')
    torch.backends.quantized.engine = 'x86'
    torch.ao.quantization.quantize_dtype = torch.qint8
    fuse_modules(agent)
    agent_prepared = torch.ao.quantization.prepare_qat(agent.train(), inplace=False)
    agent_prepared.train()
    agent_int8 = torch.ao.quantization.convert(agent_prepared.eval(), inplace=False)
    agent_int8.load_model(f"models/qat/trpo-{config.env_name}-default-qnnpack-fuse.pt")
    agent_int8.eval()
    quant_start = time.time()
    avg_return_int8, steps_quant = agent_int8.evaluation(seed[i])
    quant_end = time.time()
    #agent_int8.save_model(f"models/qat/trpo-{config.env_name}.pt")
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


