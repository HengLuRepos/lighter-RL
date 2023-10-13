from config import *
import torch
import gymnasium as gym
from td3_quantize import TwinDelayedDDPG
import numpy as np
import time
from torch.ao.quantization.qconfig import QConfig, get_default_qat_qconfig, default_qat_qconfig, default_qat_qconfig_v2
from torch.ao.quantization.fake_quantize import default_fused_act_fake_quant,default_fused_wt_fake_quant
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, default_per_channel_weight_observer
#seed = [2,3,4,5,6,7,8,9,10,11]
seed = [2]
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
    agent = TwinDelayedDDPG(env, config)
    agent.load_model(f"models/TD3-{config.env_name}-seed-1.pt")
    agent = agent.to('cpu')
    origin_start = time.time()
    agent.eval()
    avg_return, steps_origin = agent.evaluation(seed=seed[i])
    #avg_return, steps_origin = agent.evaluation()
    origin_end = time.time()

    agent.eval()
    agent.qconfig = get_default_qat_qconfig(backend='qnnpack')
    torch.backends.quantized.engine = 'qnnpack'
    torch.ao.quantization.quantize_dtype = torch.qint8
    #fuse_modules(agent)
    agent_prepared = torch.ao.quantization.prepare_qat(agent.to('cuda:1').train(), inplace=False)
    agent_prepared.train()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    state, _ = agent_prepared.env.reset(seed=config.seed)
    done = False
    for step in range(config.max_timestamp):
        episode_timesteps += 1
        if step < config.start_steps:
            action = agent_prepared.actor(torch.as_tensor(state, dtype=torch.float, device=agent_prepared.device)).detach().cpu().numpy()
        else:
            action = agent_prepared.actor.explore(torch.as_tensor(state, dtype=torch.float, device=agent_prepared.device)).detach().cpu().numpy()
        next_state, reward, terminated, truncated, _ = agent_prepared.env.step(action)
        done = terminated or truncated
        agent_prepared.buffer.remember(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if step >= config.start_steps:
            agent_prepared.train_iter()
        if done:
            print(f"TD3-{config.env_name}-{torch.backends.quantized.engine} Total T: {step+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            state, _ = agent_prepared.env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    agent_int8 = torch.ao.quantization.convert(agent_prepared.eval().to('cpu'), inplace=False)
    agent_int8.eval()
    # quant_start = time.time()
    # avg_return_int8, steps_quant = agent_int8.evaluation(seed=seed[i])
    # quant_end = time.time()
    agent_int8.save_model(f"models/qat/TD3-{config.env_name}-default-{torch.backends.quantized.engine}.pt")
    # fp32_time.append(origin_end - origin_start)
    # int8_time.append(quant_end - quant_start)
    # fp32_return.append(avg_return)
    # int8_return.append(avg_return_int8)
    # fp32_step.append(steps_origin)
    # int8_step.append(steps_quant)

