#!/bin/bash
#SBATCH -J llava13                  # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=120G
#SBATCH -o Report-%J.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=stekin6@gatech.edu        # E-mail address for notifications
cd  $HOME/scratch/rl-vqa                            # Change to working directory
conda init
conda activate v3fusion
python sft_llava.py
