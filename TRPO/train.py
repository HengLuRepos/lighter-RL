from trpo import TRPO
from config import *
import gymnasium as gym

config = HumanoidConfig(1)
env = gym.make(config.env)
agent = TRPO(env, config).to('cuda:0')
agent.train_agent()
