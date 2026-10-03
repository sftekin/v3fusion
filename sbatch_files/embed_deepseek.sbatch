#!/bin/bash
#SBATCH -J visual_embed_deepseek                # Job name
#SBATCH -N1 --gres=gpu:A100:2
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=120G
#SBATCH -o sbatch_files/reports/%x_%A_%a.out                  # Combined output and error messages file
#SBATCH --array=0-7                              # 2 models x 4 tasks
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=stekin6@gatech.edu        # E-mail address for notifications
cd $HOME/scratch/v3fusion                          # Change to working directory

source ~/.bashrc
conda activate deepseek

MODELS=(deepseek-vl2-small)
TASKS=(mmmu okvqa mmmu_pro ocr)

# SLURM_ARRAY_TASK_ID -> (model, task) pair: task varies fastest
NUM_TASKS=${#TASKS[@]}
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_TASKS ))
TASK_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_TASKS ))
MODEL_NAME=deepseek-vl2-small
TASK_NAME=${TASKS[$TASK_IDX]}

echo "Array task $SLURM_ARRAY_TASK_ID -> model=$MODEL_NAME task=$TASK_NAME"
python inference_scripts/obtain_visual_embeddings.py \
    --model_name "$MODEL_NAME" \
    --task_name "$TASK_NAME" \
    --dataset_type validation 
