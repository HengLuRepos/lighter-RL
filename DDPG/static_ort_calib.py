import onnxruntime as ort
import torch 
import gymnasium as gym
from config import *
from ddpg import DDPG
import numpy as np
import time
import argparse
import psutil
from onnxruntime.quantization import CalibrationDataReader, quantize_static
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
env = gym.make(config.env)
origin_session = ort.InferenceSession(f"models/DDPG-{config.env_name}-seed-1.onnx", providers=ort.get_available_providers())

all_steps = 0
states = []
state, _ = env.reset()
states.append(state)
while all_steps < 10000:
    input_name = origin_session.get_inputs()[0].name
    action = origin_session.run(None, {input_name: state[None,:].astype(np.float32)})[0]
    state, reward, terminated, truncated, _ = env.step(action[0])
    states.append(state)
    all_steps += 1
    if terminated or truncated:
        state, _ = env.reset()
        states.append(state)
states = np.array(states,dtype=np.float32)

class QuntizationDataReader(CalibrationDataReader):
    def __init__(self, states, batch_size, input_name):

        self.torch_dl = torch.as_tensor(states, dtype=torch.float)

        self.input_name = input_name
        self.datasize = len(self.torch_dl)

        self.enum_data = iter(self.torch_dl)

    def to_numpy(self, pt_tensor):
        return pt_tensor.detach().cpu().numpy() if pt_tensor.requires_grad else pt_tensor.cpu().numpy()

    def get_next(self):
        batch = next(self.enum_data, None)
        if batch is not None:
          return {self.input_name: self.to_numpy(batch)[None,:]}
        else:
          return None

    def rewind(self):
        self.enum_data = iter(self.torch_dl)

qdr = QuntizationDataReader(states, batch_size=64, input_name=origin_session.get_inputs()[0].name)
q_static_opts = {"ActivationSymmetric":False,
                 "WeightSymmetric":True}

quantized_model = quantize_static(model_input=f"models/onnxQuant/DDPG-{config.env_name}-prep.onnx", 
                                  model_output=f"models/onnxQuant/DDPG-{config.env_name}-static.onnx",
                                  calibration_data_reader=qdr,
                                  extra_options=q_static_opts)
session = ort.InferenceSession(f"models/onnxQuant/DDPG-{config.env_name}-static.onnx", providers=ort.get_available_providers())
input_name = session.get_inputs()[0].name
state, info = env.reset(seed=seed[0]+100)
with torch.no_grad():
    for i in range(100):
        origin_start = time.time()
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
            origin_end = time.time()
        fp32_return.append(r)
        fp32_time.append(origin_end - origin_start)
        fp32_step.append(step)
        fp32_ram.append(psutil.Process().memory_info().rss / (1024 * 1024))
    print(f"{np.mean(fp32_return):.2f},{np.std(fp32_return):.2f},{np.mean(fp32_time):.2f},{np.std(fp32_time):.2f},{np.mean(fp32_step):.2f},{np.std(fp32_step):.2f},{np.mean(fp32_ram):.2f},{np.std(fp32_ram):.2f}")

