from ppo_quantize import PPO
from config import CartPoleConfig, PendulumConfig, HalfCheetahConfig
import gymnasium as gym
import torch
import time

env = gym.make("CartPole-v1")
config = PendulumConfig()
ppo = PPO(env, config, 2, device='cpu')
ppo.load_model("./models/ppo-256-CartPole-best.pt")
ppo.eval()

origin_start = time.time()
ppo.evaluation()
origin_end = time.time()
ppo_int8 = torch.ao.quantization.quantize_dynamic(
    ppo, 
    {torch.nn.Linear},
    dtype=torch.qint8)


print("after quant")
quant_start = time.time()
ppo_int8.evaluation()
quant_end = time.time()
print("fp32 eval time: {:.2f}".format(origin_end-origin_start))
print("int8 eval time: {:.2f}".format(quant_end-quant_start))
ppo_int8.save_model("./models-ppo-qint8.pt")

