from config import *
import torch
import gymnasium as gym
from ddpg_quantize import DDPG
import numpy as np
import time
from torch.ao.quantization.qconfig import QConfig, get_default_qat_qconfig
import argparse
import psutil
env_map = {
    "HalfCheetah-v4": HalfCheetahConfig,
    "Humanoid-v4": HumanoidConfig,
    "HumanoidStandup-v4": HumanoidStandupConfig,
    "Ant-v4": AntConfig,
    "Hopper-v4": HopperConfig,

}
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4",
        help="the id of the environment")
    args = parser.parse_args()
    
    return args
args = parse_args()
cfg = env_map[args.env_id]
seed = [2,3,4,5,6,7,8,9,10,11]
int8_time = []
int8_step = []
int8_return = []
int8_ram = []
config = cfg(seed[0])
env = gym.make(config.env)
agent = DDPG(env, config).to('cpu')
#agent.load_model(f"models/DDPG-{config.env_name}-seed-1.pt")
agent.eval()
agent.qconfig = get_default_qat_qconfig(backend='x86')
torch.backends.quantized.engine = 'x86'
torch.ao.quantization.quantize_dtype = torch.qint8
agent_prepared = torch.ao.quantization.prepare_qat(agent.train(), inplace=False)
agent_prepared.train()
agent_int8 = torch.ao.quantization.convert(agent_prepared.eval(), inplace=False)
del agent, agent_prepared
agent_int8.load_model(f"models/qat/DDPG-{config.env_name}-default-x86.pt")
agent_int8.eval()
state, info = env.reset(seed=seed[0]+100)
torch.onnx.export(agent_int8, torch.as_tensor(state[None,:], dtype=torch.float), f"models/onnxQuant/DDPG-{config.env}-qat.onnx")