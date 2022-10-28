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

#ContextNet Append NOT PERFORMED
#argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att author --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds graph --append append --resume  ./Models/kgm_author_best_model.pth.tar >salidas/kgm_author_out.txt 2>salidas/author_error.txt"
#argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att school --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds graph --append append --resume  ./Models/kgm_school_best_model.pth.tar >salidas/kgm_school_out.txt 2>salidas/school_error.txt"
#argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att time --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds graph --append append --resume ./Models/kgm_time_best_model.pth.tar >salidas/kgm_time_out.txt 2>salidas/time_error.txt"
#argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att type --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds graph --append append --resume ./Models/kgm_type_best_model.pth.tar >salidas/kgm_type_out.txt 2>salidas/type_error.txt"

# BoW + FCM
#argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att author --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds bow --resume ./Models/fcm_bow_author_best_model.pth.tar >salidas/fcm_author_out.txt 2>salidas/author_error.txt"
#argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att school --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds bow --resume ./Models/fcm_bow_school_best_model.pth.tar >salidas/fcm_school_out.txt 2>salidas/school_error.txt"
#argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att time --dir_dataset   ../SemArt/ --batch_size 128 --nepochs 300 --embedds bow --resume ./Models/fcm_bow_time_best_model.pth.tar >salidas/fcm_time_out.txt 2>salidas/time_error.txt"
# argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att type --dir_dataset   ../SemArt/ --batch_size 128 --nepochs 300 --embedds bow --resume ./Models/fcm_bow_type_best_model.pth.tar >salidas/fcm_type_out.txt 2>salidas/type_error.txt"

# TF-IDF + FCM
#argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att author --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds tfidf --resume ./Models/fcm_tf_author_best_model.pth.tar >salidas/author_out.txt 2>salidas/author_error.txt"
#argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att school --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds tfidf --resume ./Models/fcm_tf_school_best_model.pth.tar >salidas/school_out.txt 2>salidas/school_error.txt"
#argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att time   --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds tfidf --resume ./Models/fcm_tf_time_best_model.pth.tar >salidas/time_out.txt 2>salidas/time_error.txt"
#argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att type   --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds tfidf --resume ./Models/fcm_tf_type_best_model.pth.tar >salidas/type_out.txt 2>salidas/type_error.txt"

# TF-IDF + FRBC
#argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att author --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds frbc --clusters 39"
#argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att school --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds frbc --clusters 39"
#argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att time   --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds frbc --clusters 39"
#argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att type   --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds frbc --clusters 39"

# TF-IDF + FRBC
argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --append append --att author --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds clip --clusters 39"
argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --append append --att school --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds clip --clusters 39"
argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --append append --att time   --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds clip --clusters 39"
argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --append append --att type   --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds clip --clusters 39"

# BoW + FCM append
#argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --append append --att author --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds bow --resume  ./Models/bow_append_author_best_model.pth.tar >salidas/author_out_append.txt 2>salidas/author_error_append.txt"
#argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --append append --att school --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds bow --resume  ./Models/fcm_bow_append_school_best_model.pth.tar >salidas/school_out_append.txt 2>salidas/school_error_append.txt"
#argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --append append --att time --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds bow --resume  ./Models/fcm_bow_append_time_best_model.pth.tar >salidas/time_out_append.txt 2>salidas/time_error_append.txt"
#argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --append append --att type --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds bow --resume ./Models/fcm_bow_append_type_best_model.pth.tar >salidas/type_out_append.txt 2>salidas/type_error_append.txt"


# TF-IDF + FCM append
#argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --append append --att author --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds tfidf  --resume ./Models/fcm_tf_append_author_best_model.pth.tar >salidas/tfauthor_append_out.txt 2>salidas/tfauthor_append_error.txt"
#argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --append append --att school --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds tfidf  --resume ./Models/fcm_tf_append_school_best_model.pth.tar >salidas/tfschool_append_out.txt 2>salidas/tfschool_append_error.txt"
#argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --append append --att time   --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds tfidf  --resume ./Models/fcm_tf_append_time_best_model.pth.tar >salidas/tftime_append_out.txt 2>salidas/tftime_append_error.txt"


# MTL
#argumentos[${#argumentos[@]}]="--mode train --workers 0 --model mtl --append gradient --att all   --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --architecture resnet  --resume ./Models/vgg_all_best_model.pth.tar"
#argumentos[${#argumentos[@]}]="--mode train --workers 0 --model mtl --append gradient --att all   --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --architecture resnet  --resume ./Models/vgg_all_best_model.pth.tar"

#argumentos[${#argumentos[@]}]="--mode train --workers 0 --model kgm --att author --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds bow --k 100 --append append

srun python main.py ${argumentos[SLURM_ARRAY_TASK_ID-1]}

