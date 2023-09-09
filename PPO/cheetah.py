from config import HalfCheetahConfig
import gymnasium as gym
from ppo import PPO
config = HalfCheetahConfig()
env = gym.make("HalfCheetah-v4")
ppo = PPO(env, config, 42)
ppo.train("HalfCheetah")