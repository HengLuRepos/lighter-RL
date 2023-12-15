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
fig.suptitle(f"{args.algs} l2(dim1) avg return")
fig.supxlabel("prune amount")
fig.supylabel("scaled average return")
envs = ["HalfCheetah-v4", "HumanoidStandup-v4", "Ant-v4"]
methods = ("Baseline", "PTDQ", "PTSQ", "QAT")
for idx, env in enumerate(envs):
  path = f"{args.algs}/csv/{env}.csv"
  data = pd.read_csv(path, header=None)
  x = data[0].values
  avg = data[1].values
  avg_std = data[2].values
  avg_std /= avg[0]
  avg /= avg[0]
  
  axs[idx].plot(x, avg)
  axs[idx].errorbar(x, avg, yerr=avg_std)
fig.legend()
plt.savefig(f"figs/l2-dim1/{args.algs}-l2(dim1)-pruning-avg-return.png")
