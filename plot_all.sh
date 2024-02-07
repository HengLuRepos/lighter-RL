#!/bin/bash

folders=("DDPG" "TD3" "SAC-cleanrl" "PPO-cleanrl" "TRPO")


envs=("HalfCheetah-v4" "Ant-v4" "HumanoidStandup-v4" "Hopper-v4" "Humanoid-v4")

for folder in "${folders[@]}"; do
    python plot.py --algs "$folder" --n 1
    python plot.py --algs "$folder" --n 2
done
