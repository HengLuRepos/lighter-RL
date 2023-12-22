#!/bin/bash

folders=("DDPG" "TD3" "SAC-cleanrl" "PPO-cleanrl" "TRPO")


envs=("Humanoid-v4" "Hopper-v4")

for folder in "${folders[@]}"; do
	if [ -d "$folder" ]; then
		cd "${folder}"

		for env in "${envs[@]}"; do
            python baseline.py --env-id "$env" >> "csv/quantization/${env}.csv"
            sleep 15
            python ptdq.py --env-id "$env" >> "csv/quantization/${env}.csv"
            sleep 15
            python ptsq.py --env-id "$env" >> "csv/quantization/${env}.csv"
            sleep 15
            python eval_qat.py --env-id "$env" >> "csv/quantization/${env}.csv"
            sleep 15
		done
		cd ..
	fi
done
