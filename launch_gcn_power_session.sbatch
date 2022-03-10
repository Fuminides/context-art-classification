#!/bin/bash
#SBATCH --partition=power_std
#SBATCH --account=acc_ure_power_std
#SBATCH --gres=gpu:v100:1
#SBATCH --array=1-4
# Activate the conda environment named "pytorch"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
# Move to the src/runs/ folder
#cd ../src/runs
# Set all the tasks to perform
argumentos[${#argumentos[@]}]="--mode train --workers 0 --model gcn --dir_dataset "../SemArt/" --batch_size 126 --nepochs 300 --embedds gcn --resume "./Models/fcm_author_best_model.pth.tar" >salidas/author_out.txt --resume "./Models/gcn_author_best_model.pth.tar" 2>salidas/gcn_error.txt"


srun python main.py ${argumentos[SLURM_ARRAY_TASK_ID-1]}
