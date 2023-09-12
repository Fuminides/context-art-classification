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


# CLIP context 1 problema con la loss, 2 problema con la memoria, 3 ok, 4 ok de momento
argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att all --dir_dataset ../SemArt/ --architecture vit --batch_size 256 --nepochs 300 --embedds bow --resume ./Models/vit_fcm_best_model.pth.tar >salidas/train_vit.txt" 
argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att all --dir_dataset ../SemArt/ --architecture convnext --batch_size 256 --nepochs 300 --embedds bow --resume ./Models/convnext_fcm_best_model.pth.tar >salidas/train_convnext.txt" 

argumentos[${#argumentos[@]}]="--mode test --workers 0 --model kgm --att all --dir_dataset ../SemArt/ --architecture convnext --batch_size 256 --nepochs 300 --embedds bow --model_path ./Models/convnext_fcm_best_model.pth.tar >salidas/test_vit.txt" 
argumentos[${#argumentos[@]}]="--mode test --workers 0 --model kgm --att all --dir_dataset ../SemArt/ --architecture vit --batch_size 256 --nepochs 300 --embedds bow --resume ./Models/vit_fcm_best_model.pth.tar >salidas/test_convnext.txt" 


srun python main.py ${argumentos[SLURM_ARRAY_TASK_ID-1]}

