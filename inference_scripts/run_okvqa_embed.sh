#!/bin/bash

# Default values (can be overridden by command-line arguments)
TASK_NAME="okvqa"
DATASET_TYPE="validation"
NUM_SAMPLES=1500

# Define the list of models
# models=("llava-v1.6-vicuna-7b-hf" "llava-v1.6-vicuna-13b-hf" \
#        "Qwen2.5-VL-7B-Instruct" "InternVL2-8B")

models=("deepseek-vl2-small" "deepseek-vl2-tiny")


for model in "${models[@]}"; do
    echo "Running model: $model on task: $TASK_NAME split: $DATASET_TYPE"
    python inference_scripts/obtain_visual_embeddings.py \
        --model_name "$model" \
        --task_name "$TASK_NAME" \
        --dataset_type "$DATASET_TYPE" \
        --num_samples "$NUM_SAMPLES"
done
