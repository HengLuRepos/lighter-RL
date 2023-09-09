from ppo import PPO
from config import CartPoleConfig, PendulumConfig
import gymnasium as gym

env = gym.make("CartPole-v1")
config = CartPoleConfig()
ppo = PPO(env, config, 42)
ppo.train("CartPole")

