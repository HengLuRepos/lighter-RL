#!/bin/bash

folders=("DDPG" "TD3" "SAC-cleanrl" "PPO-cleanrl" "TRPO")


envs=("HalfCheetah-v4" "HumanoidStandup-v4" "Ant-v4")

for folder in "${folders[@]}"; do
	if [ -d "$folder" ]; then
		cd "${folder}"

		for env in "${envs[@]}"; do
			python baseline.py --env-id "$env" >> output.log
			sleep 60
			for amount in 0.05 0.1 0.15 0.2 0.25 0.3; do
				python pruning.py --env-id "$env" --prune-amount "$amount" >> output.log
				sleep 60
			done
		done
		cd ..
	fi
done
