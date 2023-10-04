from sac import SAC
import gymnasium as gym
from config import *
import torch
config = HumanoidStandupConfig(1)
env = gym.make(config.env)
agent = SAC(env, config).to('cuda:0')
agent.train_agent()
