from sac import SAC
import gymnasium as gym
from config import *
import torch
config = Walker2DConfig(1)
env = gym.make(config.env)
agent = SAC(env, config).to('cuda:1')
agent.train_agent()
