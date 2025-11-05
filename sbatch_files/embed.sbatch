#!/bin/bash
#SBATCH -J visual_embed                 # Job name
#SBATCH -N1 --gres=gpu:A100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=120G
#SBATCH -o reports/%x_%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=stekin6@gatech.edu        # E-mail address for notifications
cd $HOME/scratch/rl-vqa                            # Change to working directory

# conda activate llamas
# python inference_scripts/obtain_visual_embeddings.py --model_name Qwen2.5-VL-7B-Instruct --dataset_type validation
# python inference_scripts/obtain_visual_embeddings.py --model_name llava-v1.6-vicuna-7b-hf --dataset_type validation
# python inference_scripts/obtain_visual_embeddings.py --model_name InternVL2-8B --dataset_type validation

conda activate deepseek
python inference_scripts/obtain_visual_embeddings.py --model_name deepseek-vl2-small --task_name mmmu
python inference_scripts/obtain_visual_embeddings.py --model_name deepseek-vl2-small --task_name okvqa