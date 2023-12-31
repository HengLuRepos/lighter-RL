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
    path = f"{args.algs}/csv/{file}-l1.csv"
    data = pd.read_csv(path, header=None)
    x = data[0].values
    avg = data[1].values
    avg_std = data[2].values
    avg_std /= avg[0]
    avg /= avg[0]
    time = data[3].values
    time_std = data[4].values
    time_std /= time[0]
    time /= time[0]
    leng = data[5].values
    leng_std = data[6].values
    leng_std /= leng[0]
    leng /= leng[0]
    ram = data[7].values
    ram /= ram[0]
    plt.figure()
    plt.plot(x, avg)
    #plt.fill_between(x, avg - avg_std, avg + avg_std, alpha=0.5)
    plt.errorbar(x, avg, yerr=avg_std)
    plt.xlabel("pruning amount")
    plt.ylabel("scaled average return")
    plt.title(f"{file}-return")
    plt.savefig(f"{args.algs}/figs/{args.algs}-{file}-l1-scaled return.png")
    plt.figure()
    plt.plot(x, time)
    #plt.fill_between(x, time - time_std, time + time_std, alpha=0.5)
    plt.errorbar(x, time, yerr=time_std)
    plt.xlabel("pruning amount")
    plt.ylabel("inference time")
    plt.title(f"{file}-time")
    plt.savefig(f"{args.algs}/figs/{args.algs}-{file}-l1-scaled inference time.png")
    plt.figure()
    plt.plot(x, leng)
    #plt.fill_between(x, leng - leng_std, leng + leng_std, alpha=0.5)
    plt.errorbar(x, leng, yerr=leng_std)
    plt.xlabel("pruning amount")
    plt.ylabel("episode length")
    plt.title(f"{file}-ep-length")
    plt.savefig(f"{args.algs}/figs/{args.algs}-{file}-l1-scaled ep length.png")
    plt.figure()
    plt.plot(x, ram)
    plt.xlabel("pruning amount")
    plt.ylabel("ram usage")
    plt.title(f"{file}-ram")
    plt.savefig(f"{args.algs}/figs/{args.algs}-{file}-l1-scaled ram usage.png")


    
