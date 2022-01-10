#!/bin/bash
#$ -q cal.q
#$ -cwd
source activate py365

python main.py --mode train --model kgm --att school --dir_dataset "../SemArt/" --batch_size 126 --nepochs 300 --embedds fcm >salidas/school_out.txt 2>salidas/school_error.txt