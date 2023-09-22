import torch
from torch.profiler import profile, record_function, ProfilerActivity
from config import *
import torch
import gymnasium as gym
from trpo import TRPO
import numpy as np
import time
from torch.ao.quantization.qconfig import QConfig, get_default_qat_qconfig
from torch.ao.quantization.fake_quantize import default_fused_act_fake_quant,default_fused_wt_fake_quant
seed = [2]
config = AntConfig(seed[0])
env = gym.make(config.env)
agent = TRPO(env, config).to('cpu')
state, _ = env.reset(seed = config.seed + 100)
def fuse_modules(model):
    if hasattr(model, 'fuse_modules'):
        model.fuse_modules()
    for p in list(model.modules())[1:]:
        fuse_modules(p)
agent.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
torch.backends.quantized.engine = 'x86'
torch.ao.quantization.quantize_dtype = torch.qint8
fuse_modules(agent)
agent_prepared = torch.ao.quantization.prepare_qat(agent.train(), inplace=False)
agent_prepared.train()
agent_int8 = torch.ao.quantization.convert(agent_prepared.eval(), inplace=False)
agent_int8.load_model(f"models/qat/trpo-{config.env_name}-default-qnnpack.pt")
#prof = profile(activities=[
#        ProfilerActivity.CPU, ProfilerActivity.CUDA], 
#        record_shapes=True, 
#        with_stack=True,
#        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{config.env_name}-qnnpack'))
#prof.start()
#for s in seed:
#    agent_int8.evaluation(seed=s)
#prof.stop()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True) as prof:
    agent_int8.evaluation(seed=seed[0])
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
