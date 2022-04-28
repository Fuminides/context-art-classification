import os

import pandas as pd

def load_semart_symbols(args_dict):
    # Load data
    if args_dict.mode == 'train':
        textfile = os.path.join(args_dict.dir_dataset, args_dict.csvtrain)
    elif args_dict.mode == 'val':
        textfile = os.path.join(args_dict.dir_dataset, args_dict.csvval)
    elif args_dict.mode == 'test':
        textfile = os.path.join(args_dict.dir_dataset, args_dict.csvtest)
    
    df = pd.read_csv(textfile, delimiter='\t', encoding='Cp1252')
    imagefolder = os.path.join(args_dict.dir_dataset, args_dict.dir_images)

    