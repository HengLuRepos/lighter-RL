from ppo import PPO
from config import CartPoleConfig, PendulumConfig
import gymnasium as gym

env = gym.make("InvertedPendulum-v4")
config = PendulumConfig()
ppo = PPO(env, config, 42)
ppo.train("InvertedPendulum")

