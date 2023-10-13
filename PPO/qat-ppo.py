from config import *
import torch
import gymnasium as gym
from ppo_quantize import PPO
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
    config = HalfCheetahConfig(seed[i])
    env = gym.make(config.env)
    agent = PPO(env, config)
    agent.load_model(f"models/ppo-{config.env_name}-seed-1-best.pt")
    agent = agent.to('cpu')
    origin_start = time.time()
    agent.eval()
    #avg_return, steps_origin = agent.evaluation(seed=seed[i])
    #avg_return, steps_origin = agent.evaluation()
    origin_end = time.time()

    agent.eval()
    agent.qconfig = get_default_qat_qconfig(backend='qnnpack')
    torch.backends.quantized.engine = 'qnnpack'
    torch.ao.quantization.quantize_dtype = torch.qint8
    #fuse_modules(agent)
    agent_prepared = torch.ao.quantization.prepare_qat(agent.to('cuda:1').train(), inplace=False)
    agent_prepared.train()
    for ep in range(10 * config.epoch):
        paths, episodic_rewards = agent_prepared.sample_batch()
        states = np.concatenate([path["states"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        rewards = np.concatenate([path["rewards"] for path in paths])
        next_states = np.concatenate([path["next_states"] for path in paths])
        old_logprobs = np.concatenate([path["log_probs"] for path in paths])
        returns = agent_prepared.get_returns(paths)

        states = torch.as_tensor(states, dtype=torch.float, device='cuda:1')
        next_states = torch.as_tensor(next_states, dtype=torch.float, device='cuda:1')
        advantages = agent_prepared.baseline.calc_advantage(states, next_states, rewards)
        for _ in range(agent_prepared.config.update_freq):
            agent_prepared.update_policy(states, actions, old_logprobs, advantages)
            agent_prepared.update_baseline(returns, states)
        avg_reward = np.mean(episodic_rewards)
        print(f"Iter {ep} Avg reward {avg_reward:.3f}")
    agent_int8 = torch.ao.quantization.convert(agent_prepared.eval().to('cpu'), inplace=False)
    agent_int8.eval()
    # quant_start = time.time()
    # avg_return_int8, steps_quant = agent_int8.evaluation(seed=seed[i])
    # quant_end = time.time()
    agent_int8.save_model(f"models/qat/ppo-{config.env_name}-default-{torch.backends.quantized.engine}-best.pt")
    # fp32_time.append(origin_end - origin_start)
    # int8_time.append(quant_end - quant_start)
    # fp32_return.append(avg_return)
    # int8_return.append(avg_return_int8)
    # fp32_step.append(steps_origin)
    # int8_step.append(steps_quant)


