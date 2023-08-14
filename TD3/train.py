from td3 import TwinDelayedDDPG
from config import PusherConfig, ReacherConfig, SwimmerConfig
import gymnasium as gym

config = ReacherConfig(1)
env = gym.make(config.env)
agent = TwinDelayedDDPG(env, config)
agent.train_agent()