from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import argparse
# Parallel environments

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4",
        help="the id of the environment")
    args = parser.parse_args()
    
    return args
args = parse_args()
vec_env = make_vec_env(args.env_id, n_envs=1)
model = A2C("MlpPolicy", vec_env, device="cuda:0", seed=1)
model.learn(total_timesteps=1000000)
model.save(f"models/a2c-{args.env_id}-seed-1.pt")