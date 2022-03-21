'''

Generates node2vec embeddings for SemArt.

@author: Javier Fumanal Idocin

'''
from scipy.sparse import coo_matrix, csr_matrix, dok_matrix

import pandas as pd
import numpy as np

def generate_kg_semart(semart_path=r'/home/javierfumanal/Documents/GitHub/SemArt'):
    '''
    Returns a sparse matrix that contains the KG for the SEMART dataset and a dict
    containing the index for each label.
    '''
    # Generate connections category
    def generate_connects(coo_matrix_fill, df, category, key_dict):
        unique_elems = np.unique(train_df[category])
        
        for element in unique_elems:
             connected = df[category] == element
             indexes_nonzero = np.nonzero(np.array(connected))[0]
             
             for idx_nz in indexes_nonzero:
                 coo_matrix_fill[key_dict[element], idx_nz] = 1
                 coo_matrix_fill[idx_nz, key_dict[element]] = 1
        
        return coo_matrix_fill
            
    train_df = pd.read_csv(semart_path + r'/semart_train.csv', sep='\t', encoding='latin1')
    # test_df = pd.read_csv(semart_path + r'/semart_test.csv', sep='\t')
    
    authors = np.unique(train_df['AUTHOR'])
    school = np.unique(train_df['SCHOOL'])
    timeframe = np.unique(train_df['TIMEFRAME'])
    types = np.unique(train_df['TYPE'])
    
    extra_nodes = len(authors) + len(school) + len(timeframe) + len(types)
    
    kg_coo = dok_matrix((extra_nodes, train_df.shape[0] + extra_nodes), dtype=np.int8)
    
    
    all_categories = np.concatenate((authors, school, timeframe, types))
    key_dicts = {x:ix+train_df.shape[0] for ix, x in enumerate(all_categories)}
    generate_connects(kg_coo, train_df, 'AUTHOR', key_dicts)
    generate_connects(kg_coo, train_df, 'SCHOOL', key_dicts)
    generate_connects(kg_coo, train_df, 'TIMEFRAME', key_dicts)
    generate_connects(kg_coo, train_df, 'TYPE', key_dicts)
    
    return  kg_coo, key_dicts


def save_edges(matrix_to_save, res_file='kg_semart.csv'):
    '''
    Saves semart kg graph in a file format ready for node2vec.

    Returns:
        None.

    '''
    cx = coo_matrix(matrix_to_save)
    
    with open(res_file, 'w') as f:
        for i,j,v in zip(cx.row, cx.col, cx.data):
            f.write(str(i) + ' ' + str(j) + '\n')

def save_keys(key_dict, res_file='kg_keys.csv'):
    '''
    Saves Semart KeyDict in a csv format
    
    '''
    
    with open(res_file, 'w') as f:
        for key, elem in key_dict.items():
            f.write(str(key.replace(' ', '_')) + ' ' + str(elem) + '\n')
    
if __name__ == '__main__':
    kg_semart, keys_semart = generate_kg_semart()
    
    save_edges(kg_semart)
    save_keys(keys_semart)
    
    
    
    
    
    
    
                     