#!/bin/bash
#SBATCH --partition=power_std
#SBATCH --account=acc_ure_power_std
#SBATCH --gres=gpu:v100:1
#SBATCH --array=1-2
# Activate the conda environment named "pytorch"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

argumentos[${#argumentos[@]}]="--mode train --workers 0 --model mtl --architecture resnet --append gradient --att all --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --model_path ./Models/mtl_symbol_model.pth.tar --symbol_task true"
argumentos[${#argumentos[@]}]="--mode train --workers 0 --model mtl --architecture clip --append gradient --att all --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --model_path ./Models/mtl_symbol_model.pth.tar --symbol_task true"

srun python main.py ${argumentos[SLURM_ARRAY_TASK_ID-1]}
