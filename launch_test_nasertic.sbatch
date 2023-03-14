#!/bin/bash
#SBATCH --partition=power_std
#SBATCH --account=acc_ure_power_std
#SBATCH --gres=gpu:v100:1
#SBATCH --array=1-8
# Activate the conda environment named "pytorch"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
# Move to the src/runs/ folder
#cd ../src/runs

# CLIP context 
argumentos[${#argumentos[@]}]="--mode test --workers 0 --model kgm  --att author --dir_dataset ../SemArt/ --batch_size 32 --nepochs 300 --embedds clip --clusters 77 --model_path ./Models/kgm-author/clip_author_gradient_best_model.pth.tar  >salidas/author_out_clip.txt 2>salidas/author_error_clip.txt" 
argumentos[${#argumentos[@]}]="--mode test --workers 0 --model kgm  --att school --dir_dataset ../SemArt/ --batch_size 32 --nepochs 300 --embedds clip --clusters 77 --model_path ./Models/kgm-school/clip_school_gradient_best_model.pth.tar  >salidas/school_out_clip.txt 2>salidas/school_error_clip.txt"
argumentos[${#argumentos[@]}]="--mode test --workers 0 --model kgm  --att time   --dir_dataset ../SemArt/ --batch_size 32 --nepochs 300 --embedds clip --clusters 77 --model_path ./Models/kgm-time/clip_time_gradient_best_model.pth.tar    >salidas/time_out_clip.txt 2>salidas/time_error_clip.txt"
argumentos[${#argumentos[@]}]="--mode test --workers 0 --model kgm  --att type   --dir_dataset ../SemArt/ --batch_size 32 --nepochs 300 --embedds clip --clusters 77 --model_path ./Models/kgm-type/clip_type_gradient_best_model.pth.tar    >salidas/type_out_clip.txt 2>salidas/type_error_clip.txt"

# BoW + FCM
argumentos[${#argumentos[@]}]="--mode test --workers 0 --model kgm --att author --dir_dataset ../SemArt/ --batch_size 32 --nepochs 300 --embedds bow --model_path  ./Models/kgm-author/bow_author_gradient_best_model.pth.tar >salidas/author_out_fcm.txt 2>salidas/author_error_fcm.txt"
argumentos[${#argumentos[@]}]="--mode test --workers 0 --model kgm --att school --dir_dataset ../SemArt/ --batch_size 32 --nepochs 300 --embedds bow --model_path  ./Models/kgm-school/bow_school_gradient_best_model.pth.tar >salidas/school_out_fcm.txt 2>salidas/school_error_fcm.txt"
argumentos[${#argumentos[@]}]="--mode test --workers 0 --model kgm --att time --dir_dataset ../SemArt/ --batch_size 32 --nepochs 300 --embedds bow --model_path  ./Models/kgm-time/bow_time_gradient_best_model.pth.tar >salidas/time_out_fcm.txt 2>salidas/time_error_fcm.txt"
argumentos[${#argumentos[@]}]="--mode test --workers 0 --model kgm --att type --dir_dataset ../SemArt/ --batch_size 32 --nepochs 300 --embedds bow --model_path ./Models/kgm-type/bow_type_gradient_best_model.pth.tar >salidas/type_out_fcm.txt 2>salidas/type_error_fcm.txt"


srun python main.py ${argumentos[SLURM_ARRAY_TASK_ID-1]}

