from sac_quantize import SAC
import gymnasium as gym
from config import *
config = HumanoidStandupConfig(1)
env = gym.make(config.env)
agent = SAC(env, config)
agent.retrain_agent()