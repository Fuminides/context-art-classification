#!/bin/bash
#$ -q cal.q
#$ -cwd
source activate py365

python main.py --mode train --model kgm --att author --dir_dataset "../SemArt/" --batch_size 126 --nepochs 300 --embedds fcm >salidas/author_out.txt 2>salidas/author_error.txt