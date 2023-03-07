#!/bin/bash
#SBATCH --partition=power_std
#SBATCH --account=acc_ure_power_std
#SBATCH --gres=gpu:v100:1
#SBATCH --array=1-2
# Activate the conda environment named "pytorch"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
# Move to the src/runs/ folder
#cd ../src/runs
# Set all the tasks to perform


# CLIP context 1 problema con la loss, 2 problema con la memoria, 3 ok, 4 ok de momento
argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm  --att all --dir_dataset ../SemArt/ --batch_size 32 --nepochs 300 --embedds clip --clusters 39 --resume ./Models/clip_author_best_model.pth.tar  >salidas/author_out_clip.txt 2>salidas/author_error_clip.txt" 
argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att author --dir_dataset ../SemArt/ --batch_size 32 --nepochs 300 --embedds bow --resume  ./Models/bow_all_best_model.pth.tar > salidas/all_out_fcm.txt 2>salidas/all_error_fcm.txt"




srun python main.py ${argumentos[SLURM_ARRAY_TASK_ID-1]}

