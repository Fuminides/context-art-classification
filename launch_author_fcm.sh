#!/bin/bash
#$ -N Semart_FCM_school
#$ -q cal.q
#$ -cwd
#$ -t 1
#$ -e salidas/school_error.txt
#$ -o salidas/school_out.txt
source activate py365
python main.py --mode train --model kgm --att author --dir_dataset "../SemArt/" --batch_size 126 --nepochs 300 --embedds fcm