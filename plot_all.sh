#!/bin/bash

folders=("DDPG" "TD3" "SAC-cleanrl" "PPO-cleanrl" "TRPO")


envs=("Hopper-v4" "Humanoid-v4")

for folder in "${folders[@]}"; do
    python plot.py --algs "$folder" 
done
