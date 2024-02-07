#!/bin/bash

folders=("DDPG" "TD3" "SAC-cleanrl" "PPO-cleanrl" "TRPO")


envs=("HalfCheetah-v4" "Ant-v4" "HumanoidStandup-v4" "Hopper-v4" "Humanoid-v4")

for folder in "${folders[@]}"; do
    rm "$folder/figs/*.png"
    mkdir "$folder/figs/l1"
    mkdir "$folder/figs/l2"
    for env in "${envs[@]}"; do
        python plot_prun.py --algs "$folder" --env-id "$env" --n 1
        python plot_prun.py --algs "$folder" --env-id "$env" --n 2
    done
done
