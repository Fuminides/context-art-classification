import os
import pickle

import pandas as pd
import numpy as np

import params



## FUNCTIONS RELATED TO THE CIRLOT DATASET
def load_dictionary_df():
    dict_df = pd.read_csv('Data/cirlot_tabular.csv')
    
    return dict_df

def load_terms():
    return load_dictionary_df()['TERM']

def load_definitions():
    return load_dictionary_df()['DEFINITION']

def load_cirlot_as_dict():
    terms = load_dictionary_df()['TERM']
    defs = load_dictionary_df()['DEFINITION']

    return {term: def0 for term, def0 in zip(terms, defs) }

## FUNCTIONS RELATED TO THE SEMART DATA

def __load_semart_proxy(mode='train'):
    class dummyPlug: #EVA 01: YURI, DONT DO THIS TO ME!
        def __init__(self):
            pass

    args_dict = dummyPlug()
    args_dict.mode = mode
    args_dict.dir_dataset = r'G:\Mi unidad\Code\SemArt'
    args_dict.csvtrain = args_dict.dir_dataset + '\semart_train.csv'

    return load_semart_symbols(args_dict)

def load_semart_symbols(args_dict):
    # Load data
    if args_dict.mode == 'train':
        textfile = os.path.join(args_dict.dir_dataset, args_dict.csvtrain)
    elif args_dict.mode == 'val':
        textfile = os.path.join(args_dict.dir_dataset, args_dict.csvval)
    elif args_dict.mode == 'test':
        textfile = os.path.join(args_dict.dir_dataset, args_dict.csvtest)
    
    symbol_canon_list = load_terms()

    df = pd.read_csv(textfile, delimiter='\t', encoding='Cp1252')
    names = df['IMAGE_FILE']
    descriptions = df['DESCRIPTION'] # Load the contextual annotations

    dictionary_painting_symbol = np.zeros((df.shape[0], len(symbol_canon_list)))
    for ix, description in enumerate(descriptions):
        
        symbols_painting = np.zeros((len(symbol_canon_list),))

        for jx, symbol in enumerate(symbol_canon_list):
            symbol = symbol.lower()
            description = description.lower()

            if symbol in description:
                symbols_painting[jx] = 1

        dictionary_painting_symbol[ix, :] = symbols_painting
    
    return dictionary_painting_symbol, names, symbol_canon_list

def symbol_connectivity():
    from Data import symbol_graph

    trial = symbol_graph.generate_adjacency_df_symbol(symmetry=True)

    return trial

def symbol_report(symbol_matrix, symbol_names=None, painting_names=None):

    res = {}
    #Number of paintings with at least one symbol
    symbol_presence = np.mean(symbol_context.sum(axis=1)>0)
    res['symbol_paintings'] = symbol_presence

    #Number of paintings with at more than one symbol
    symbol_abundance = np.mean(symbol_context.sum(axis=1)>1)
    res['useful_paintings'] = symbol_abundance

    #Histogram of number of symbols per painting
    symbol_histogram = symbol_context.sum(axis=1)
    res['painting_histogram'] = symbol_histogram

    #Paintings sortered with more symbols
    symbol_histogram = np.argsort(symbol_context.sum(axis=1))
    if painting_names is None:
        res['important_paintings'] = symbol_histogram
    else:
        res['important_paintings'] = painting_names[symbol_histogram]

    #Symbols that appear at least one time
    symbol_histogram = np.mean(symbol_context.sum(axis=0) > 0)
    res['useful_symbols'] = symbol_histogram

    #Histogram of number of symbols
    symbol_histogram = symbol_context.sum(axis=1) > 0
    res['symbol_histogram'] = symbol_histogram

    #Most common symbols
    symbol_histogram = np.argsort(symbol_context.sum(axis=0))
    if symbol_names is None:
        res['sorted_symbols'] = symbol_histogram
    else:
        res['sorted_symbols'] = symbol_names[symbol_histogram]


    return res

    

if __name__ == '__main__':
    symbol_context, paintings_names, symbols_names = __load_semart_proxy(mode='train')
    res = symbol_report(symbol_context, symbol_names=symbols_names, painting_names=paintings_names)
    print(res)
    print('hOLA')