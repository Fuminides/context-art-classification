#!/bin/bash
#SBATCH --partition=power_std
#SBATCH --account=acc_ure_power_std
#SBATCH --gres=gpu:v100:1
#SBATCH --array=1-5
# Activate the conda environment named "pytorch"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
# Move to the src/runs/ folder
#cd ../src/runs
# Set all the tasks to perform

#ContextNet Append
argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att author --dir_dataset ../SemArt/ --batch_size 126 --nepochs 300 --embedds graph --resume --append append ./Models/fcm_author_best_model.pth.tar >salidas/author_out.txt 2>salidas/author_error.txt"
argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att school --dir_dataset ../SemArt/ --batch_size 126 --nepochs 300 --embedds graph --resume --append append ./Models/fcm_school_best_model.pth.tar >salidas/school_out.txt 2>salidas/school_error.txt"
argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att time --dir_dataset ../SemArt/ --batch_size 126 --nepochs 300 --embedds graph --resume --append append ./Models/fcm_time_best_model.pth.tar >salidas/time_out.txt 2>salidas/time_error.txt"
argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att type --dir_dataset ../SemArt/ --batch_size 126 --nepochs 300 --embedds graph --append append --resume ./Models/fcm_type_best_model.pth.tar >salidas/type_out.txt 2>salidas/type_error.txt"


srun python main.py ${argumentos[SLURM_ARRAY_TASK_ID-1]}
