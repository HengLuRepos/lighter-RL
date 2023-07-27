from ppo_quantize import PPO
from config import CartPoleConfig, PendulumConfig, HalfCheetahConfig
import gymnasium as gym
import torch
import time

env_name = "HalfCheetah"
env = gym.make("HalfCheetah-v4")
config = PendulumConfig()
ppo = PPO(env, config, 2, device='cpu')
ppo.fuse_model()
ppo.qconfig = torch.quantization.get_default_qconfig('x86')  
torch.backends.quantized.engine = 'x86'
torch.quantization.quantize_dtype = torch.qint8
torch.quantization.prepare(ppo,inplace=True)
ppo_int8 = torch.quantization.convert(ppo)
ppo_int8.load_model("./models/ppo-{}-PTSQ.pt".format(env_name))

quant_start = time.time()
ppo_int8.evaluation()
quant_end = time.time()
print("Inference time: {:.2f}".format(quant_end - quant_start))