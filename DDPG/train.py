from ddpg import DDPG
from config import *
import gymnasium as gym

config = InvertedDoublePendulumConfig(1)
env = gym.make(config.env)
agent = DDPG(env, config).to('cuda:1')
agent.train_agent()