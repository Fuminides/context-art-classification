#!/bin/bash
#$ -q cal.q
#$ -cwd
source activate py365

python main.py --mode train --model gcn --att all --dir_dataset "../SemArt/" --batch_size 126 --nepochs 300 --embedds graph --lamb_c 0.9 --lambda_e 0.1 --resume "./Models/gcn_school_best_model.pth.tar" >salidas/gcn_all.txt 2>salidas/gcn_error.txt