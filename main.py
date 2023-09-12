import warnings
import os
import utils

warnings.filterwarnings("ignore")

import torch

import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix

from dataloader_mtl import ArtDatasetMTL
#from model_gcn import NODE2VEC_OUTPUT

from params import get_parser
from train import run_train, load_att_class
from semart_test import run_test


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        args = '--mode train --workers 0 --model kgm --att all --dir_dataset ../SemArt/ --architecture vit --batch_size 1 --nepochs 300 --embedds bow --resume ./Models/fcm_bow_author_best_model.pth.tar >salidas/fcm_author_out.txt 2>salidas/author_error.txt'
        
    # Load parameters
    parser = get_parser()
    args_dict, unknown = parser.parse_known_args(args=args.split())

    assert args_dict.att in ['type', 'school', 'time', 'author', 'all'], \
        'Incorrect classifier. Please select type, school, time, author or all.'


    if args_dict.model == 'mtl':
        args_dict.att = 'all'
    args_dict.name = '{}-{}'.format(args_dict.model, args_dict.att)

    opts = vars(args_dict)
    print('------------ Options -------------')
    for k, v in sorted(opts.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-----------------------------------')

    # Check mode and model are correct
    assert args_dict.mode in ['train', 'test'], 'Incorrect mode. Please select either train or test.'
    assert args_dict.model in ['mtl', 'kgm'], 'Incorrect model. Please select either mlt, kgm, gcn or fcm.'

    # Run process
    if args_dict.mode == 'train':
        run_train(args_dict)
    elif args_dict.mode == 'test':
        run_test(args_dict)

