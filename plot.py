import pandas as pd 
import matplotlib.pyplot as plt
import argparse
import numpy as np
from matplotlib.ticker import FuncFormatter
def format_y(value, tick_number):
    return f'{value:.2f}'
formatter = FuncFormatter(format_y)

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

plt.rcParams['font.size'] = 16
envs = ["HalfCheetah-v4", "HumanoidStandup-v4", "Ant-v4", "Hopper-v4", "Humanoid-v4"]
methods = ("Baseline", "PTDQ", "PTSQ", "QAT")


fig, axs = plt.subplots(5, sharex=True)
#fig.suptitle(f"{args.algs} l{args.n} avg return")
#fig.supxlabel("prune amount")
#fig.supylabel("scaled average return")
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
  axs[idx].yaxis.set_major_formatter(formatter)
fig.subplots_adjust(hspace=0.6,left=0.1,right=0.95)
plt.savefig(f"figs/l{args.n}/return/{args.algs}-l{args.n}-pruning-avg-return.png",dpi=600)

fig, axs = plt.subplots(5, sharex=True)
#fig.suptitle(f"{args.algs} l{args.n} avg return")
#fig.supxlabel("prune amount")
#fig.supylabel("scaled average return")
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
  axs[idx].yaxis.set_major_formatter(formatter)
fig.subplots_adjust(hspace=0.6,left=0.1,right=0.95)
plt.savefig(f"figs/l{args.n}/infer/{args.algs}-l{args.n}-pruning-infer-time.png",dpi=600)

fig, axs = plt.subplots(5, sharex=True)
#fig.suptitle(f"{args.algs} l{args.n} avg return")
#fig.supxlabel("prune amount")
#fig.supylabel("scaled average return")
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
  axs[idx].yaxis.set_major_formatter(formatter)
fig.subplots_adjust(hspace=0.6,left=0.1,right=0.95)
plt.savefig(f"figs/l{args.n}/len/{args.algs}-l{args.n}-pruning-ep-len.png",dpi=600)

fig, axs = plt.subplots(5, sharex=True)
#fig.suptitle(f"{args.algs} l{args.n} avg return")
#fig.supxlabel("prune amount")
#fig.supylabel("scaled average return")
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
  axs[idx].yaxis.set_major_formatter(formatter)
fig.subplots_adjust(hspace=0.6,left=0.1,right=0.95)
plt.savefig(f"figs/l{args.n}/ram/{args.algs}-l{args.n}-pruning-ram.png",dpi=600)

fig, axs = plt.subplots(5, sharex=True)
#fig.suptitle(f"{args.algs} l{args.n} avg return")
#fig.supxlabel("prune amount")
#fig.supylabel("scaled average return")
for idx, env in enumerate(envs):
  path = f"{args.algs}/csv/{env}-l{args.n}.csv"
  data = pd.read_csv(path, header=None)
  x = data[0].values
  avg = data[9].values

  avg /= np.abs(avg[0])
  
  axs[idx].plot(x, avg)
  axs[idx].set_title(env)
  axs[idx].errorbar(x, avg, yerr=avg_std)
  axs[idx].yaxis.set_major_formatter(formatter)
fig.subplots_adjust(hspace=0.6,left=0.1,right=0.95)
plt.savefig(f"figs/l{args.n}/energy/{args.algs}-l{args.n}-pruning-energy.png",dpi=600)

"""

fig, axs = plt.subplots(5, sharex=True)

for idx, env in enumerate(envs):
  path = f"{args.algs}/csv/onnxQuant/{env}.csv"
  data = pd.read_csv(path, header=None)
  avg = data[2].values
  avg_std = data[3].values
  avg_std /= np.abs(avg[0])
  avg /= np.abs(avg[0])
  
  axs[idx].bar(methods, avg, yerr=avg_std)
  axs[idx].set_title(env)
fig.subplots_adjust(hspace=0.6,left=0.05,right=0.95)
#fig.legend()
plt.savefig(f"figs/onnxQuant/infer/{args.algs}-infer.png",dpi=600)

fig, axs = plt.subplots(5, sharex=True)

for idx, env in enumerate(envs):
  path = f"{args.algs}/csv/onnxQuant/{env}.csv"
  data = pd.read_csv(path, header=None)
  avg = data[0].values
  avg_std = data[1].values
  avg_std /= np.abs(avg[0])
  avg /= np.abs(avg[0])
  
  axs[idx].bar(methods, avg, yerr=avg_std)
  axs[idx].set_title(env)
fig.subplots_adjust(hspace=0.6,left=0.05,right=0.95)
#fig.legend()
plt.savefig(f"figs/onnxQuant/return/{args.algs}-return.png",dpi=600)

fig, axs = plt.subplots(5, sharex=True)

for idx, env in enumerate(envs):
  path = f"{args.algs}/csv/onnxQuant/{env}.csv"
  data = pd.read_csv(path, header=None)
  avg = data[4].values
  avg_std = data[5].values
  avg_std /= np.abs(avg[0])
  avg /= np.abs(avg[0])
  
  axs[idx].bar(methods, avg, yerr=avg_std)
  axs[idx].set_title(env)
fig.subplots_adjust(hspace=0.6,left=0.05,right=0.95)
#fig.legend()
plt.savefig(f"figs/onnxQuant/len/{args.algs}-length.png",dpi=600)

fig, axs = plt.subplots(5, sharex=True)

for idx, env in enumerate(envs):
  path = f"{args.algs}/csv/onnxQuant/{env}.csv"
  data = pd.read_csv(path, header=None)
  avg = data[6].values
  avg_std = data[7].values
  avg_std /= np.abs(avg[0])
  avg /= np.abs(avg[0])
  
  axs[idx].bar(methods, avg, yerr=avg_std)
  axs[idx].set_title(env)
fig.subplots_adjust(hspace=0.6,left=0.05,right=0.95)
#fig.legend()
plt.savefig(f"figs/onnxQuant/ram/{args.algs}-ram.png",dpi=600)

fig, axs = plt.subplots(5, sharex=True)

for idx, env in enumerate(envs):
  path = f"{args.algs}/csv/onnxQuant/{env}.csv"
  data = pd.read_csv(path, header=None)
  avg = data[8].values

  avg /= np.abs(avg[0])
  
  axs[idx].bar(methods, avg, yerr=avg_std)
  axs[idx].set_title(env)
fig.subplots_adjust(hspace=0.6,left=0.05,right=0.95)
#fig.legend()
plt.savefig(f"figs/onnxQuant/energy/{args.algs}-energy.png",dpi=600)

"""