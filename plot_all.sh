#!/bin/bash

folders=("DDPG-cleanrl" "TD3-cleanrl")


envs=("HalfCheetah-v4" "HumanoidStandup-v4" "Ant-v4")

for folder in "${folders[@]}"; do
    for env in "${envs[@]}"; do
        python plot_prun.py --env-id "$env" --algs "$folder"
    done
done
