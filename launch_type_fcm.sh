#!/bin/bash
#$ -N Semart_FCM_school
#$ -q cal.q
#$ -cwd
#$ -t 1
#$ -e school_error.txt
#$ -o school_out.txt
python main.py --mode train --model kgm --att type --dir_dataset "../SemArt/" --batch_size 126 --nepochs 300 --embedds fcm