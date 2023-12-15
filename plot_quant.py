import pandas as pd 
import matplotlib.pyplot as plt
import argparse
import numpy as np
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4",
        help="the id of the environment")
    parser.add_argument("--algs", type=str, default="DDPG",
        help="the id of the environment")
    args = parser.parse_args()
    
    return args
args = parse_args()
methods = ("Baseline", "PTDQ", "PTSQ", "QAT")
file = f"{args.algs}/csv/quantization/{args.env_id}.csv"
data = pd.read_csv(file, header=None)
avg_return = data[0].values
avg_return_std = data[1].values
infer, infer_std = data[2].values, data[3].values
ep_len, ep_len_std = data[4].values, data[5].values
energy = data[6].values
ram, ram_std = data[7].values, data[8].values
model_size = data[9].values
if model_size[0] < model_size[1]:
    model_size[0] *= 1024
x = np.arange(len(methods))
plt.figure()
plt.bar(methods, avg_return, yerr=avg_return_std)
plt.ylabel("average return")
plt.title(f"Quantization {args.algs} {args.env_id} average return")
plt.savefig(f"{args.algs}/figs/quantization/{args.algs}-{args.env_id}-avg-return.png")

plt.figure()
plt.bar(methods, infer, yerr=infer_std)
plt.ylabel("inference time")
plt.title(f"Quantization {args.algs} {args.env_id} inference time")
plt.savefig(f"{args.algs}/figs/quantization/{args.algs}-{args.env_id}-infer-time.png")

plt.figure()
plt.bar(methods, ep_len, yerr=ep_len_std)
plt.ylabel("episodic length")
plt.title(f"Quantization {args.algs} {args.env_id} episodic length")
plt.savefig(f"{args.algs}/figs/quantization/{args.algs}-{args.env_id}-length.png")

plt.figure()
plt.bar(methods, ram, yerr=ram_std)
plt.ylabel("ram usage")
plt.title(f"Quantization {args.algs} {args.env_id} ram usage")
plt.savefig(f"{args.algs}/figs/quantization/{args.algs}-{args.env_id}-ram.png")

plt.figure()
plt.bar(methods, energy)
plt.ylabel("energy")
plt.title(f"Quantization {args.algs} {args.env_id} energy")
plt.savefig(f"{args.algs}/figs/quantization/{args.algs}-{args.env_id}-energy.png")

plt.figure()
plt.bar(methods, model_size)
plt.ylabel("model size")
plt.title(f"Quantization {args.algs} {args.env_id} model size")
plt.savefig(f"{args.algs}/figs/quantization/{args.algs}-{args.env_id}-model-size.png")