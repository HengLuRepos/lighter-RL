from a2c_online import A2C
from config import *
import gymnasium as gym

env = gym.make("HalfCheetah-v4")
config = HalfCheetahConfig(1)
agent = A2C(env, config).to("cuda")
agent.load_model(f"models/a2c-HalfCheetah-seed-1-best.pt")
agent.evaluation()
