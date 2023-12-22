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

fig, axs = plt.subplots(5, sharex=True)
fig.suptitle(f"{args.algs} l1 avg return")
fig.supxlabel("prune amount")
fig.supylabel("scaled average return")
envs = ["HalfCheetah-v4", "HumanoidStandup-v4", "Ant-v4", "Hopper-v4", "Humanoid-v4"]
methods = ("Baseline", "PTDQ", "PTSQ", "QAT")
for idx, env in enumerate(envs):
  path = f"{args.algs}/csv/{env}-l1.csv"
  data = pd.read_csv(path, header=None)
  x = data[0].values
  avg = data[1].values
  avg_std = data[2].values
  avg_std /= avg[0]
  avg /= avg[0]
  
  axs[idx].plot(x, avg)
  axs[idx].set_title(env)
  axs[idx].errorbar(x, avg, yerr=avg_std)
fig.subplots_adjust(hspace=0.6)
fig.legend()
plt.savefig(f"figs/l1/{args.algs}-l1-pruning-avg-return.png")
