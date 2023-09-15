from config import *
import torch
import gymnasium as gym
from trpo import TRPO, np2torch
import numpy as np
import time
def evaluation(ag, config):
    env = gym.make(config.env)
    ep_reward = 0
    state, _ = env.reset(seed = config.seed + 100)
    steps = 0
    for i in range(config.eval_epochs):
        state, _ = env.reset()
        done = False
        while not done:
            action = ag(np2torch(state[None,:]).to('cpu')).detach().cpu().numpy().squeeze(axis=0)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            steps += 1
    print("---------------------------------------")
    print(f"Evaluation over {config.eval_epochs} episodes: {ep_reward/config.eval_epochs:.3f}")
    print("---------------------------------------")
    return ep_reward/config.eval_epochs, steps/config.eval_epochs
seed = [2,3,4,5,6,7,8,9,10,11]
fp32_time = []
int8_time = []
fp32_step = []
int8_step = []
fp32_return = []
int8_return = []
def fuse_modules(model):
    if hasattr(model, 'fuse_modules'):
        model.fuse_modules()
    for p in list(model.modules())[1:]:
        fuse_modules(p)
for i in range(len(seed)):
    config = AntConfig(seed[i])
    env = gym.make(config.env)
    agent = TRPO(env, config).to('cpu')
    agent.load_model(f"models/TRPO-{config.env_name}-seed-1.pt")
    origin_start = time.time()
    avg_return, steps_origin = evaluation(agent, config)
    origin_end = time.time()

    agent.eval()
    agent.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    torch.backends.quantized.engine = 'x86'
    torch.ao.quantization.quantize_dtype = torch.qint8
    fuse_modules(agent)
    agent_prepared = torch.ao.quantization.prepare(agent)
    env_temp = gym.make(config.env)
    state, _ = env_temp.reset(seed=seed[i] + 100)
    for _ in range(1000):
        agent_prepared(np2torch(state[None,:]).to('cpu'))
        state = env_temp.observation_space.sample()
    agent_int8 = torch.ao.quantization.convert(agent_prepared)
    
    quant_start = time.time()
    avg_return_int8, steps_quant = evaluation(agent_int8, config)
    quant_end = time.time()
    agent_int8.save_model(f"models/static_quantize/TRPO-{config.env_name}.pt")
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

