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
    parser.add_argument("--n", type=int, default=1)
    args = parser.parse_args()
    
    return args
args = parse_args()


envs = ["HalfCheetah-v4", "HumanoidStandup-v4", "Ant-v4", "Hopper-v4", "Humanoid-v4"]
methods = ("Baseline", "PTDQ", "PTSQ", "QAT")


fig, axs = plt.subplots(5, sharex=True)
fig.suptitle(f"{args.algs} l{args.n} avg return")
fig.supxlabel("prune amount")
fig.supylabel("scaled average return")
for idx, env in enumerate(envs):
  path = f"{args.algs}/csv/{env}-l{args.n}.csv"
  data = pd.read_csv(path, header=None)
  x = data[0].values
  avg = data[1].values
  avg_std = data[2].values
  avg_std /= np.abs(avg[0])
  avg /= np.abs(avg[0])
  
  axs[idx].plot(x, avg)
  axs[idx].set_title(env)
  axs[idx].errorbar(x, avg, yerr=avg_std)
fig.subplots_adjust(hspace=0.6)
fig.legend()
plt.savefig(f"figs/l{args.n}/{args.algs}-l{args.n}-pruning-avg-return.png")

fig, axs = plt.subplots(5, sharex=True)
fig.suptitle(f"{args.algs} l{args.n} inference time")
fig.supxlabel("prune amount")
fig.supylabel("scaled inference time")
for idx, env in enumerate(envs):
  path = f"{args.algs}/csv/{env}-l{args.n}.csv"
  data = pd.read_csv(path, header=None)
  x = data[0].values
  avg = data[3].values
  avg_std = data[4].values
  avg_std /= np.abs(avg[0])
  avg /= np.abs(avg[0])
  
  axs[idx].plot(x, avg)
  axs[idx].set_title(env)
  axs[idx].errorbar(x, avg, yerr=avg_std)
fig.subplots_adjust(hspace=0.6)
fig.legend()
plt.savefig(f"figs/l{args.n}/{args.algs}-l{args.n}-pruning-infer-time.png")

fig, axs = plt.subplots(5, sharex=True)
fig.suptitle(f"{args.algs} l{args.n} episodic length")
fig.supxlabel("prune amount")
fig.supylabel("scaled episodic length")
for idx, env in enumerate(envs):
  path = f"{args.algs}/csv/{env}-l{args.n}.csv"
  data = pd.read_csv(path, header=None)
  x = data[0].values
  avg = data[5].values
  avg_std = data[6].values
  avg_std /= np.abs(avg[0])
  avg /= np.abs(avg[0])
  
  axs[idx].plot(x, avg)
  axs[idx].set_title(env)
  axs[idx].errorbar(x, avg, yerr=avg_std)
fig.subplots_adjust(hspace=0.6)
fig.legend()
plt.savefig(f"figs/l{args.n}/{args.algs}-l{args.n}-pruning-ep-len.png")

fig, axs = plt.subplots(5, sharex=True)
fig.suptitle(f"{args.algs} l{args.n} ram usage")
fig.supxlabel("prune amount")
fig.supylabel("scaled ram usage")
for idx, env in enumerate(envs):
  path = f"{args.algs}/csv/{env}-l{args.n}.csv"
  data = pd.read_csv(path, header=None)
  x = data[0].values
  avg = data[7].values
  avg_std = data[8].values
  avg_std /= np.abs(avg[0])
  avg /= np.abs(avg[0])
  
  axs[idx].plot(x, avg)
  axs[idx].set_title(env)
  axs[idx].errorbar(x, avg, yerr=avg_std)
fig.subplots_adjust(hspace=0.6)
fig.legend()
plt.savefig(f"figs/l{args.n}/{args.algs}-l{args.n}-pruning-ram.png")

fig, axs = plt.subplots(5, sharex=True)
fig.suptitle(f"{args.algs} l{args.n} energy usage")
fig.supxlabel("prune amount")
fig.supylabel("scaled energy usage")
for idx, env in enumerate(envs):
  path = f"{args.algs}/csv/{env}-l{args.n}.csv"
  data = pd.read_csv(path, header=None)
  x = data[0].values
  avg = data[9].values

  avg /= np.abs(avg[0])
  
  axs[idx].plot(x, avg)
  axs[idx].set_title(env)
  axs[idx].errorbar(x, avg, yerr=avg_std)
fig.subplots_adjust(hspace=0.6)
fig.legend()
plt.savefig(f"figs/l{args.n}/{args.algs}-l{args.n}-pruning-energy.png")

"""
fig, axs = plt.subplots(5, sharex=True)
fig.suptitle(f"{args.algs} onnx inference time")
fig.supylabel("scaled inference time")
for idx, env in enumerate(envs):
  path = f"{args.algs}/onnxQuant/{env}.csv"
  data = pd.read_csv(path, header=None)
  avg = data[2].values
  avg_std = data[3].values
  avg_std /= np.abs(avg[0])
  avg /= np.abs(avg[0])
  
  axs[idx].bar(methods, avg)
  axs[idx].set_title(env)
  axs[idx].errorbar(methods, avg, yerr=avg_std)
fig.subplots_adjust(hspace=0.6)
fig.legend()
plt.savefig(f"figs/onnxQuant/infer/{args.algs}-infer.png")

fig, axs = plt.subplots(5, sharex=True)
fig.suptitle(f"{args.algs} onnx average return")
fig.supylabel("scaled average return")
for idx, env in enumerate(envs):
  path = f"{args.algs}/onnxQuant/{env}.csv"
  data = pd.read_csv(path, header=None)
  avg = data[0].values
  avg_std = data[1].values
  avg_std /= np.abs(avg[0])
  avg /= np.abs(avg[0])
  
  axs[idx].bar(methods, avg)
  axs[idx].set_title(env)
  axs[idx].errorbar(methods, avg, yerr=avg_std)
fig.subplots_adjust(hspace=0.6)
fig.legend()
plt.savefig(f"figs/onnxQuant/return/{args.algs}-return.png")

fig, axs = plt.subplots(5, sharex=True)
fig.suptitle(f"{args.algs} onnx episodic length")
fig.supylabel("scaled episodic length")
for idx, env in enumerate(envs):
  path = f"{args.algs}/onnxQuant/{env}.csv"
  data = pd.read_csv(path, header=None)
  avg = data[4].values
  avg_std = data[5].values
  avg_std /= np.abs(avg[0])
  avg /= np.abs(avg[0])
  
  axs[idx].bar(methods, avg)
  axs[idx].set_title(env)
  axs[idx].errorbar(methods, avg, yerr=avg_std)
fig.subplots_adjust(hspace=0.6)
fig.legend()
plt.savefig(f"figs/onnxQuant/len/{args.algs}-length.png")

fig, axs = plt.subplots(5, sharex=True)
fig.suptitle(f"{args.algs} onnx ram usage")
fig.supylabel("scaled ram usage")
for idx, env in enumerate(envs):
  path = f"{args.algs}/onnxQuant/{env}.csv"
  data = pd.read_csv(path, header=None)
  avg = data[6].values
  avg_std = data[7].values
  avg_std /= np.abs(avg[0])
  avg /= np.abs(avg[0])
  
  axs[idx].bar(methods, avg)
  axs[idx].set_title(env)
  axs[idx].errorbar(methods, avg, yerr=avg_std)
fig.subplots_adjust(hspace=0.6)
fig.legend()
plt.savefig(f"figs/onnxQuant/ram/{args.algs}-ram.png")

fig, axs = plt.subplots(5, sharex=True)
fig.suptitle(f"{args.algs} onnx energy")
fig.supylabel("scaled energy")
for idx, env in enumerate(envs):
  path = f"{args.algs}/onnxQuant/{env}.csv"
  data = pd.read_csv(path, header=None)
  avg = data[8].values

  avg /= np.abs(avg[0])
  
  axs[idx].bar(methods, avg)
  axs[idx].set_title(env)
fig.subplots_adjust(hspace=0.6)
fig.legend()
plt.savefig(f"figs/onnxQuant/energy/{args.algs}-energy.png")

"""