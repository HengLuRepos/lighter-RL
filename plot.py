import pandas as pd 
import matplotlib.pyplot as plt
import argparse
import numpy as np

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    # Algorithm specific arguments
    parser.add_argument("--algs", type=str, default="DDPG",
        help="the id of the environment")
    args = parser.parse_args()
    
    return args
args = parse_args()

fig, axs = plt.subplots(3, sharex=True)
fig.suptitle(f"{args.algs} quantization model size")
fig.supylabel("model size")
envs = ["HalfCheetah-v4", "HumanoidStandup-v4", "Ant-v4"]
methods = ("Baseline", "PTDQ", "PTSQ", "QAT")
for idx, env in enumerate(envs):
  path = f"{args.algs}/csv/quantization/{env}.csv"
  data = pd.read_csv(path, header=None)
  model_size = data[9].values
  if model_size[0] < model_size[1]:
    model_size[0] *= 1024
  axs[idx].bar(methods, model_size)
fig.legend()
plt.savefig(f"figs/quant/size/{args.algs}-quantization-model-size.png")
