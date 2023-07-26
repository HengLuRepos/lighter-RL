import torch
from utils import device
from sac import SAC
from config import Config
import gymnasium as gym
import numpy as np
env = gym.make("InvertedPendulum-v4")
config = Config()
sac = SAC(env, config, 1)
sac.train()
    

