#!/bin/bash
#$ -q cal.q
#$ -cwd
source activate py365

python main.py --mode train --model kgm --att time --dir_dataset "../SemArt/" --batch_size 126 --nepochs 300 --embedds fcm --resume "./Models/fcm_time_best_model.pth.tar" >salidas/time_out.txt 2>salidas/time_error.txt