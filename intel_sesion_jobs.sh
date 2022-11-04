#!/bin/bash

source ~/anaconda3Intel/etc/profile.d/conda.sh
conda activate py39

tsp -S 4
# python --mode train --workers 0 --model kgm --append gradient --att author --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds bow --clusters 128 --k 100 # Already done
# python --mode train --workers 0 --model kgm --append gradient --att school --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds bow --clusters 128 --k 100 # Already done
python main.py --mode train --workers 0 --model kgm --append gradient --att time   --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds bow --clusters 128 --k 100
CUDA_VISIBLE_DEVICES=1 python main.py --mode train --workers 0 --model kgm --append gradient --att type   --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds bow --clusters 128 --k 100
 
python main.py --mode train --workers 0 --model kgm --append gradient --att author --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds clip --clusters 128 --k 100
CUDA_VISIBLE_DEVICES=1 python main.py --mode train --workers 0 --model kgm --append gradient --att school --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds clip --clusters 128 --k 100
python main.py --mode train --workers 0 --model kgm --append gradient --att time   --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds clip --clusters 128 --k 100
CUDA_VISIBLE_DEVICES=1 python main.py --mode train --workers 0 --model kgm --append gradient --att type   --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds clip --clusters 128 --k 100
 
python main.py --mode train --workers 0 --model kgm --append gradient --att author --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds graph --clusters 128 --k 100
CUDA_VISIBLE_DEVICES=1 python main.py --mode train --workers 0 --model kgm --append gradient --att school --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds graph --clusters 128 --k 100
python main.py --mode train --workers 0 --model kgm --append gradient --att time   --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds graph --clusters 128 --k 100
CUDA_VISIBLE_DEVICES=1 python main.py --mode train --workers 0 --model kgm --append gradient --att type   --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds graph --clusters 128 --k 100
 
python main.py --mode train --workers 0 --model mtl --append gradient --att author --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds graph --clusters 128 --k 100
CUDA_VISIBLE_DEVICES=1 python main.py --mode train --workers 0 --model mtl --append gradient --att school --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds graph --clusters 128 --k 100
python main.py --mode train --workers 0 --model mtl --append gradient --att time   --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds graph --clusters 128 --k 100
CUDA_VISIBLE_DEVICES=1 python main.py --mode train --workers 0 --model mtl --append gradient --att type   --dir_dataset ../SemArt/ --batch_size 128 --nepochs 300 --embedds graph --clusters 128 --k 100
