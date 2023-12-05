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
files = [f"{args.env_id}", f"{args.env_id}-dim0"]
for file in files:
    path = f"{args.algs}/csv/{file}.csv"
    data = pd.read_csv(path, header=None)
    x = data[0].values
    avg = data[1].values
    avg /= avg[0]
    time = data[3].values
    leng = data[5].values
    ram = data[7].values
    plt.figure()
    plt.plot(x, avg)
    plt.xlabel("pruning amount")
    plt.ylabel("scaled average return")
    plt.savefig(f"{args.algs}/figs/{file}-return.png")
    plt.figure()
    plt.plot(x, time)
    plt.xlabel("pruning amount")
    plt.ylabel("inference time")
    plt.savefig(f"{args.algs}/figs/{file}-time.png")
    plt.figure()
    plt.plot(x, leng)
    plt.xlabel("pruning amount")
    plt.ylabel("episode length")
    plt.savefig(f"{args.algs}/figs/{file}-ep.png")
    plt.figure()
    plt.plot(x, ram)
    plt.xlabel("pruning amount")
    plt.ylabel("ram usage")
    plt.savefig(f"{args.algs}/figs/{file}-ran.png")
    
