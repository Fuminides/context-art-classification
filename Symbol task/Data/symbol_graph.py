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

def load_dict():
    try:
        dict_mix = pd.read_csv(CIRLO_DICT)
    except FileNotFoundError:
        dict_mix = pd.read_csv('Data/' + CIRLO_DICT)
    
    return dict_mix

def generate_adjacency_symbol_sparse(symmetry=True):
    '''
    Returns a sparse matrix that contains the connectivy graph between the studied symbols.
    
    :param symmetry: True if you want the resulting matrix to be symmetric.
    :returns: a sparse dok_matrix with the corresponding adjacency matrix.
    '''
    dict_mix = load_dict()
    terms = dict_mix['TERM']  
    definitions = dict_mix['DEFINITION']  

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


def generate_adjacency_symbol_sparse_reduced(terms, symmetry=True):
    dict_mix = load_dict()
    definitions_og = dict_mix['DEFINITION']  
    terms_og = dict_mix['TERM']  

    definitions = []
    for term_og, definition in zip(terms_og, definitions_og):
        if term_og in terms:
            definitions.append(definition)

    res = dok_matrix((len(terms), len(terms)), dtype=np.int8)
    for ix, term in enumerate(terms):
        term_procesed = term.lower()
        for jx, definition in enumerate(definitions):
            definition_pocesed = definition.lower()
        
            if term_procesed in definition_pocesed:
                res[jx, ix] = 1
                if symmetry:
                    res[ix, jx] = 1
                    
    return res

def generate_cirlot_edges_df():
    dict_mix = load_dict()
    definitions = dict_mix['DEFINITION']  
    terms = dict_mix['TERM']  
    res = pd.DataFrame(None, columns=['Source', 'Target'])


    for ix, term in enumerate(terms):
        term_procesed = term.lower()
        for jx, definition in enumerate(definitions):
            definition_pocesed = definition.lower()
        
            if ix != jx:
                if term_procesed in definition_pocesed.split():
                    res = pd.concat([res, pd.DataFrame.from_dict({'Source': [term_procesed], 'Target': [terms[jx].lower()]})], ignore_index=True)
                
                    
    return res
def generate_adjacency_df_symbol(symmetry=True):
    '''
    Returns a datafrme with the corresponding adjacency matrix of the mix dictionary.
    '''
    sparse_matrix = generate_adjacency_symbol_sparse(symmetry=symmetry)
    dict_mix = load_dict()
    terms = list(dict_mix.keys())
    
    dense_conectivity = sparse_matrix.todense() 
    res = pd.DataFrame(dense_conectivity, index=terms, columns=terms)
    
    return res

def load_terms():
    dict_mix = load_dict()
    terms = list(dict_mix.keys())  

    return terms


if __name__ == '__main__':
    trial = generate_adjacency_df_symbol(symmetry=True)