from ppo_quantize import PPO
from config import CartPoleConfig, PendulumConfig, HalfCheetahConfig
import gymnasium as gym
import torch
import time

env_name = "HalfCheetah"
env = gym.make("HalfCheetah-v4")
config = PendulumConfig()
ppo = PPO(env, config, 2, device='cpu')
ppo.load_model("./models/ppo-256-{}.pt".format(env_name))
ppo.eval()
ppo.fuse_model()

origin_start = time.time()
ppo.evaluation()
origin_end = time.time()

convert_cali_start = time.time()
ppo.qconfig = torch.ao.quantization.get_default_qconfig('x86')  
torch.backends.quantized.engine = 'x86'
torch.ao.quantization.quantize_dtype = torch.qint8
torch.ao.quantization.prepare(ppo,inplace=True)

cali_start = time.time()
ppo.evaluation()
cali_end = time.time()

torch.quantization.convert(ppo, inplace=True)
convert_cali_end = time.time()

ppo.eval()
quant_start = time.time()
ppo.evaluation()
quant_end = time.time()

print("fp32 eval time: {:.2f}".format(origin_end-origin_start))
print("int8 eval time: {:.2f}".format(quant_end-quant_start))
print("convert + cali time: {:.2f}".format(convert_cali_end - convert_cali_start))
print("cali time: {:.2f}".format(cali_end - cali_start))
ppo.save_model("./models/ppo-{}-PTSQ.pt".format(env_name))
