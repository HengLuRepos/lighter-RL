#!/bin/bash

folders=("DDPG" "TD3" "SAC-cleanrl" "PPO-cleanrl" "TRPO")


envs=("HalfCheetah-v4" "Ant-v4" "HumanoidStandup-v4" "Humanoid-v4" "Hopper-v4")


for folder in "${folders[@]}"; do
	if [ -d "$folder" ]; then
		cd "${folder}"

		for env in "${envs[@]}"; do
			output=$(python baseline.py --env-id "$env") 
			echo "0.0,$output">> "csv/${env}-l2.csv"
			echo "0.0,$output">> "csv/${env}-l1.csv"
			sleep 10
			for amount in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7; do
			  python prune_save.py --env-id "$env" --prune-amount "$amount" --n 2
				output=$(python pruning.py --env-id "$env" --prune-amount "$amount" --n 2) 
				echo "$amount,$output">> "csv/${env}-l2.csv"
				sleep 10
			done
			for amount in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7; do
			  python prune_save.py --env-id "$env" --prune-amount "$amount" --n 1
				output=$(python pruning.py --env-id "$env" --prune-amount "$amount" --n 1)
				echo "$amount,$output">> "csv/${env}-l1.csv"
				sleep 10
			done
		done
		cd ..
	fi
done
