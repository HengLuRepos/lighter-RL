from config import *
import gymnasium as gym
from ppo import PPO
config = AntConfig(1)
env = gym.make(config.env)
agent = PPO(env, config).to('cuda:0')
agent.train_agent()