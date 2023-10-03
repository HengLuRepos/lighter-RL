from a2c_batch import A2C
import gymnasium as gym
from config import *

config = HumanoidStandupConfig(2)
env = gym.make(config.env)
agent = A2C(env, config).to('cuda:0')
agent.train_agent()