from config import *
import torch
import gymnasium as gym
from trpo import TRPO
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
    config = InvertedDoublePendulumConfig(seed[i])
    env = gym.make(config.env)
    agent = TRPO(env, config)
    agent.load_model(f"models/trpo-{config.env_name}-seed-1.pt")
    agent = agent.to('cpu')
    origin_start = time.time()
    agent.eval()
    avg_return, steps_origin = agent.evaluation(seed=seed[i])
    #avg_return, steps_origin = agent.evaluation()
    origin_end = time.time()

    agent.eval()
    agent.qconfig = get_default_qat_qconfig(backend='x86')
    torch.backends.quantized.engine = 'x86'
    torch.ao.quantization.quantize_dtype = torch.qint8
    #fuse_modules(agent)
    agent_prepared = torch.ao.quantization.prepare_qat(agent.to('cuda:0').train(), inplace=False)
    agent_prepared.train()
    num_updates = config.max_timestamp // config.batch_size
    for ep in range(num_updates):
        paths, episodic_rewards = agent_prepared.sample_batch()
        states = np.concatenate([path["states"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        rewards = np.concatenate([path["rewards"] for path in paths])
        next_states = np.concatenate([path["next_states"] for path in paths])
        done = np.concatenate([path["done"] for path in paths])
        old_logp = np.concatenate([path["log_prob"] for path in paths])
        returns = agent_prepared.get_returns(paths)

        agent_prepared.line_search(states, actions, next_states, rewards, done, old_logp)
        agent_prepared.update_critic(returns, states)
        avg_reward = np.mean(episodic_rewards)
        print(f"Iter {ep} Avg reward {avg_reward:.3f}")
    agent_int8 = torch.ao.quantization.convert(agent_prepared.eval().to('cpu'), inplace=False)
    agent_int8.eval()
    # quant_start = time.time()
    # avg_return_int8, steps_quant = agent_int8.evaluation(seed=seed[i])
    # quant_end = time.time()
    agent_int8.save_model(f"models/qat/trpo-{config.env_name}-default-{torch.backends.quantized.engine}.pt")
    # fp32_time.append(origin_end - origin_start)
    # int8_time.append(quant_end - quant_start)
    # fp32_return.append(avg_return)
    # int8_return.append(avg_return_int8)
    # fp32_step.append(steps_origin)
    # int8_step.append(steps_quant)


