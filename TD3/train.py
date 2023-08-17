from td3 import TwinDelayedDDPG
from config import *
import gymnasium as gym

config = AntConfig(1)
env = gym.make(config.env)
agent = TwinDelayedDDPG(env, config)
agent.train_agent()