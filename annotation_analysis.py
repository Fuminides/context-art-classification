import os
import pickle

import pandas as pd
import numpy as np

import params
## FUNCTIONS RELATED TO THE ORIGINAL DATASET
def load_dictionary_definitions():
    with open('Data/validated_symbol_definitions.pckl', 'rb') as f:
        aux = pickle.load(f)
    
    return aux

## FUNCTIONS RELATED TO THE SEMART DATA

def load_semart_symbols(args_dict):
    # Load data
    if args_dict.mode == 'train':
        textfile = os.path.join(args_dict.dir_dataset, args_dict.csvtrain)
    elif args_dict.mode == 'val':
        textfile = os.path.join(args_dict.dir_dataset, args_dict.csvval)
    elif args_dict.mode == 'test':
        textfile = os.path.join(args_dict.dir_dataset, args_dict.csvtest)
    
    symbol_dictionary = load_dictionary_definitions()
    symbol_canon_list = symbol_dictionary.keys()

    df = pd.read_csv(textfile, delimiter='\t', encoding='Cp1252')

    names = df['IMAGE_FILE']
    descriptions = df['DESCRIPTION'] # Load the contextual annotations

    dictionary_painting_symbol = {}
    for ix, description in enumerate(descriptions):
        words = description.split()
        symbols_painting = np.zeros(symbol_canon_list)

        for word in words:
            if word in symbol_canon_list:
                symbols_painting[ix] = 1

        dictionary_painting_symbol[names.iloc[ix]] = symbols_painting
    
    return dictionary_painting_symbol

def symbol_connectivity():
    from Data import symbol_graph

    trial = symbol_graph.generate_adjacency_df_symbol(symmetry=True)

    return trial

if __name__ == '__main__':
    args_dict = params.get_parser()
    aux = load_semart_symbols(args_dict)