from td3 import TwinDelayedDDPG
from config import *
import gymnasium as gym

config = InvertedDoublePendulumConfig(1)
env = gym.make(config.env)
agent = TwinDelayedDDPG(env, config).to('cuda:1')
agent.train_agent()