#!/bin/bash

folders=("DDPG" "TD3" "SAC-cleanrl" "PPO-cleanrl" "TRPO")


envs=("HalfCheetah-v4" "HumanoidStandup-v4" "Ant-v4")

for folder in "${folders[@]}"; do
    for env in "${envs[@]}"; do
        python plot_quant.py --env-id "$env" --algs "$folder"
    done
done
