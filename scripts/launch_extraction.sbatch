#!/bin/bash
#SBATCH --partition=power_std
#SBATCH --account=acc_ure_power_std
#SBATCH --gres=gpu:v100:1
# Activate the conda environment named "pytorch"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
# Move to the src/runs/ folder
#cd ../src/runs
# Set all the tasks to perform


python main.py --mode train --workers 0 --model kgm --append gradient --att all --dir_dataset ../SemArt/ --batch_size 32 --nepochs 0 --embedds bow --model_path ./Models/bow_all_gradient_best_model.pth.tar --grad_cam_model_path ./Models/lenet56_model.pt --grad_cam_image_size 56
python main.py --mode test --workers 0 --model kgm --append gradient --att all --dir_dataset ../SemArt/ --batch_size 32 --nepochs 0 --embedds bow --model_path ./Models/bow_all_gradient_best_model.pth.tar --grad_cam_model_path ./Models/lenet56_model.pt --grad_cam_image_size 56