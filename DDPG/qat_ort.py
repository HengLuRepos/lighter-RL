import onnxruntime as ort
import torch 
import gymnasium as gym
from config import *
from ddpg import DDPG
import numpy as np
import time
import argparse
import psutil
env_map = {
    "HalfCheetah-v4": HalfCheetahConfig,
    "Humanoid-v4": HumanoidConfig,
    "HumanoidStandup-v4": HumanoidStandupConfig,
    "Ant-v4": AntConfig,
    "Hopper-v4": HopperConfig,

}
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4",
        help="the id of the environment")
    args = parser.parse_args()
    
    return args
args = parse_args()
cfg = env_map[args.env_id]
seed = [2]
fp32_time = []
fp32_step = []
fp32_return = []
fp32_ram = []
config = cfg(seed[0])
env = gym.make(config.env, render_mode='rgb_array')
session = ort.InferenceSession(f"models/onnxQuant/DDPG-{config.env}-qat.onnx", providers=ort.get_available_providers())
input_name = session.get_inputs()[0].name
state, info = env.reset(seed=seed[0]+100)
with torch.no_grad():
    for i in range(1):
        origin_start = time.time()
        env = gym.wrappers.RecordVideo(env=env, video_folder="video",name_prefix=f"DDPG-{config.env}-qat")
        env.start_video_recorder()
        state, _ = env.reset()
        done = False
        r = 0.0
        step = 0
        while not done:
            action = session.run(None, {input_name: state[None,:].astype(np.float32)})[0]
            state, reward, terminated, truncated, _ = env.step(action[0])
            r += reward
            done = terminated or truncated
            step += 1
        env.close_video_recorder()
        origin_end = time.time()
        fp32_return.append(r)
        fp32_time.append(origin_end - origin_start)
        fp32_step.append(step)
        fp32_ram.append(psutil.Process().memory_info().rss / (1024 * 1024))
    print(f"{np.mean(fp32_return):.2f},{np.std(fp32_return):.2f},{np.mean(fp32_time):.2f},{np.std(fp32_time):.2f},{np.mean(fp32_step):.2f},{np.std(fp32_step):.2f},{np.mean(fp32_ram):.2f},{np.std(fp32_ram):.2f}")

