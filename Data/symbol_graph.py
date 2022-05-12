#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 09:34:33 2022

@author: javier
"""
import pickle 

import numpy as np
import pandas as pd

from scipy.sparse import dok_matrix

CIRLO_DICT = './cirlot_tabular.csv'

def generate_adjacency_symbol_sparse(symmetry=True):
    '''
    Returns a sparse matrix that contains the connectivy graph between the studied symbols.
    
    :param symmetry: True if you want the resulting matrix to be symmetric.
    :returns: a sparse dok_matrix with the corresponding adjacency matrix.
    '''
    dict_mix = pd.read_csv(open(CIRLO_DICT, 'rb'))
    terms = dict_mix['TERMS']  
    definitions = dict_mix['DEFINITIONS']  

    res = dok_matrix((len(terms), len(terms)), dtype=np.int8)
    for ix, term in enumerate(terms):
        term_procesed = term.lower()
        for jx, definition in enumerate(definitions):
            
            for ind_definition in definition:
                definition_pocesed = ind_definition.lower()
                
                if term_procesed in definition_pocesed:
                    res[jx, ix] = 1
                    if symmetry:
                        res[ix, jx] = 1
                    
    return res


def generate_adjacency_df_symbol(symmetry=True):
    '''
    Returns a datafrme with the corresponding adjacency matrix of the mix dictionary.
    '''
    sparse_matrix = generate_adjacency_symbol_sparse(symmetry=symmetry)
    dict_mix = pickle.load(open(PURGED_TERMS_DICT, 'rb'))
    terms = list(dict_mix.keys())
    
    dense_conectivity = sparse_matrix.todense() 
    res = pd.DataFrame(dense_conectivity, index=terms, columns=terms)
    
    return res

def load_terms():
    dict_mix = pickle.load(open(PURGED_TERMS_DICT, 'rb'))
    terms = list(dict_mix.keys())  

    return terms

def load_semart_annotations_titles(semart_path):
    '''
    Loads for each painting the title + annotation
    '''  

if __name__ == '__main__':
    trial = generate_adjacency_df_symbol(symmetry=True)