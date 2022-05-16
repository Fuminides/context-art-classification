#!/bin/bash
#$ -q cal.q
#$ -cwd
source activate py365

python main.py --mode train --model kgm --att type --dir_dataset "../SemArt/" --batch_size 126 --nepochs 300 --embedds fcm --resume "./Models/fcm_type_best_model.pth.tar" >salidas/type_out.txt 2>salidas/type_error.txt