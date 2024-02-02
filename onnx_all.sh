#!/bin/bash

folders=("PPO-cleanrl")


envs=("Ant-v4" "HalfCheetah-v4" "HumanoidStandup-v4" "Humanoid-v4" "Hopper-v4")

for folder in "${folders[@]}"; do
	if [ -d "$folder" ]; then
		cd "${folder}"
    mkdir -p "models/onnxQuant"
    mkdir -p "csv/onnxQuant"
    algo="${folder%%-*}"
		for env in "${envs[@]}"; do
            python export_ort.py --env-id "$env"
            sleep 5
            python infer_ort.py --env-id "$env" >> "csv/onnxQuant/${env}.csv"
            sleep 5
            python -m onnxruntime.quantization.preprocess --input "models/${algo}-${env}.onnx" --output "models/onnxQuant/${algo}-${env}-prep.onnx"
            
            python dyna_ort.py --env-id "$env" >> "csv/onnxQuant/${env}.csv"
            sleep 10
            python static_ort_calib.py --env-id "$env"
            sleep 10
            python static_ort.py --env-id "$env" >> "csv/onnxQuant/${env}.csv"
            sleep 5
            python convert.py --env-id "$env"
            sleep 5
            python qat_ort.py --env-id "$env" >> "csv/onnxQuant/${env}.csv"
		done
		cd ..
	fi
done
