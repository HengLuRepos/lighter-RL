#!/bin/bash

folders=("DDPG" "TD3" "SAC-cleanrl" "PPO-cleanrl" "TRPO")


envs=("HalfCheetah-v4" "Ant-v4" "HumanoidStandup-v4" "Hopper-v4" "Humanoid-v4")

for folder in "${folders[@]}"; do
    mkdir "$folder/figs/onnxQuant"
    mkdir "$folder/figs/onnxQuant"
    for env in "${envs[@]}"; do
        python plot_quant.py --algs "$folder" --env-id "$env" 
    done
done
