import torch
from torch.profiler import profile, record_function, ProfilerActivity
from config import *
import torch
import gymnasium as gym
from trpo import TRPO
import numpy as np
import time
seed = [2]
config = AntConfig(seed[0])
env = gym.make(config.env)
agent = TRPO(env, config).to('cpu')
state, _ = env.reset(seed = config.seed + 100)
agent.load_model(f"models/trpo-{config.env_name}-seed-1.pt")
#prof = profile(
#        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#        record_shapes=True,
#        with_stack=True,
#        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{config.env_name}-original'))
# prof.start()
# for s in seed:
#     agent.evaluation(seed=s)
# prof.stop()
with profile(
       activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
       record_shapes=True,
       with_stack=True) as prof:
   agent.evaluation(seed=seed[0])
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40))
