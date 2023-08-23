from ddpg import DDPG
from config import *
import gymnasium as gym

config = Walker2DConfig(1)
env = gym.make(config.env)
agent = DDPG(env, config)
agent.train_agent()