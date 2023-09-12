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
argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att author --dir_dataset ../SemArt/ --architecture resnet --batch_size 32 --nepochs 300 --embedds kmeans --resume ./Models/kmeans_author_best_model.pth.tar >salidas/kmeans_author_resnet_train_vit.txt" 
argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att type --dir_dataset ../SemArt/ --architecture resnet --batch_size 32 --nepochs 300 --embedds kmeans --resume ./Models/kmeans_type_best_model.pth.tar >salidas/kmeans_type_resnet_train.txt" 
argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att school --dir_dataset ../SemArt/ --architecture resnet --batch_size 32 --nepochs 300 --embedds kmeans --resume ./Models/kmeans_school_best_model.pth.tar >salidas/kmeans_school_resnet_train.txt" 
argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att time --dir_dataset ../SemArt/ --architecture resnet --batch_size 32 --nepochs 300 --embedds kmeans --resume ./Models/kmeans_time_best_model.pth.tar >salidas/kmeans_time_resnet_train.txt" 

argumentos[${#argumentos[@]}]="--mode test --workers 0 --model kgm --att author --dir_dataset ../SemArt/ --architecture resnet --batch_size 32 --nepochs 300 --embedds kmeans --model_path ./Models/kmeans_author_best_model.pth.tar >salidas/kmeans_author_resnet_train_vit.txt" 
argumentos[${#argumentos[@]}]="--mode test --workers 0 --model kgm --att type --dir_dataset ../SemArt/ --architecture resnet --batch_size 32 --nepochs 300 --embedds kmeans --model_path ./Models/kmeans_type_best_model.pth.tar >salidas/kmeans_type_resnet_train.txt" 
argumentos[${#argumentos[@]}]="--mode test --workers 0 --model kgm --att school --dir_dataset ../SemArt/ --architecture resnet --batch_size 32 --nepochs 300 --embedds kmeans --model_path ./Models/kmeans_school_best_model.pth.tar >salidas/kmeans_school_resnet_train.txt" 
argumentos[${#argumentos[@]}]="--mode test --workers 0 --model kgm --att time --dir_dataset ../SemArt/ --architecture resnet --batch_size 32 --nepochs 300 --embedds kmeans --model_path ./Models/kmeans_time_best_model.pth.tar >salidas/kmeans_time_resnet_train.txt" 

argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att author --dir_dataset ../SemArt/ --architecture vgg --batch_size 32 --nepochs 300 --embedds kmeans --resume ./Models/kmeans_author_best_model.pth.tar >salidas/kmeans_author_resnet_train_vit.txt" 
argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att type --dir_dataset ../SemArt/ --architecture vgg --batch_size 32 --nepochs 300 --embedds kmeans --resume ./Models/kmeans_type_best_model.pth.tar >salidas/kmeans_type_resnet_train.txt" 
argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att school --dir_dataset ../SemArt/ --architecture vgg --batch_size 32 --nepochs 300 --embedds kmeans --resume ./Models/kmeans_school_best_model.pth.tar >salidas/kmeans_school_resnet_train.txt" 
argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att time --dir_dataset ../SemArt/ --architecture vgg --batch_size 32 --nepochs 300 --embedds kmeans --resume ./Models/kmeans_time_best_model.pth.tar >salidas/kmeans_time_resnet_train.txt" 

argumentos[${#argumentos[@]}]="--mode test --workers 0 --model kgm --att author --dir_dataset ../SemArt/ --architecture vgg --batch_size 32 --nepochs 300 --embedds kmeans --model_path ./Models/kmeans_author_best_model.pth.tar >salidas/kmeans_author_resnet_train_vit.txt" 
argumentos[${#argumentos[@]}]="--mode test --workers 0 --model kgm --att type --dir_dataset ../SemArt/ --architecture vgg --batch_size 32 --nepochs 300 --embedds kmeans --model_path ./Models/kmeans_type_best_model.pth.tar >salidas/kmeans_type_resnet_train.txt" 
argumentos[${#argumentos[@]}]="--mode test --workers 0 --model kgm --att school --dir_dataset ../SemArt/ --architecture vgg --batch_size 32 --nepochs 300 --embedds kmeans --model_path ./Models/kmeans_school_best_model.pth.tar >salidas/kmeans_school_resnet_train.txt" 
argumentos[${#argumentos[@]}]="--mode test --workers 0 --model kgm --att time --dir_dataset ../SemArt/ --architecture vgg --batch_size 32 --nepochs 300 --embedds kmeans --model_path ./Models/kmeans_time_best_model.pth.tar >salidas/kmeans_time_resnet_train.txt" 

srun python main.py ${argumentos[SLURM_ARRAY_TASK_ID-1]}

