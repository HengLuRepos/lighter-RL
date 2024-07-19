from trpo import TRPO
from config import *
import gymnasium as gym
import torch
config = HalfCheetahConfig(1)
env = gym.make(config.env)
agent = TRPO(env, config)
#agent.train_agent()
state,_ = env.reset()
agent.load_model(f"models/trpo-{config.env_name}-seed-1.pt")
torch.onnx.export(agent, torch.as_tensor(state[None,:], dtype=torch.float), f"models/TRPO-{config.env}.onnx")
agent.load_model(f"models/trpo-{config.env_name}-seed-1-best.pt")
torch.onnx.export(agent, torch.as_tensor(state[None,:], dtype=torch.float), f"models/TRPO-{config.env}-best.onnx")
