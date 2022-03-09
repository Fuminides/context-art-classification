'''

Generates node2vec embeddings for SemArt.

@author: Javier Fumanal Idocin

'''
from scipy.sparse import coo_array, csr_array

import pandas as pd
import numpy as np

def generate_kg_semart(semart_path=r'/home/javier/Documents/GitHub/SemArt'):
    '''
    Returns a sparse matrix that contains the KG for the SEMART dataset
    '''
    # Generate connections category
    def generate_connects(coo_matrix, df, category, key_dict):
        unique_elems = np.unique(train_df[category])
        
        for element in unique_elems:
             connected = df[category] == element
             
             coo_matrix[key_dict[ix], connected] = 1
        
        return coo_matrix
            
    train_df = pd.read_csv(semart_path + r'/semart_train.csv', sep='\t')
    # test_df = pd.read_csv(semart_path + r'/semart_test.csv', sep='\t')
    
    authors = np.unique(train_df['AUTHOR'])
    school = np.unique(train_df['SCHOOL'])
    timeframe = np.unique(train_df['TIMEFRAME'])
    types = np.unique(train_df['TYPE'])
    
    extra_nodes = len(authors) + len(school) + len(timeframe) + len(types)
    
    kg_coo = coo_array((train_df.shape[0] + extra_nodes, train_df.shape[0] + extra_nodes), dtype=np.int8)
    
    
    all_categories = authors + school + timeframe + types
    key_dicts = {x:ix for ix, x in enumerate(all_categories)}
    generate_connects(kg_coo, train_df, 'AUTHOR')
    generate_connects(kg_coo, train_df, 'SCHOOL')
    generate_connects(kg_coo, train_df, 'TIMEFRAME')
    generate_connects(kg_coo, train_df, 'TYPE')
    
    return  kg_coo, key_dicts
    
    
    
    
    
    
    
                     