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
    def generate_connects(coo_matrix_fill, df, category, key_dict, start):
        unique_elems = np.unique(train_df[category])
        
        for element in unique_elems:
             connected = df[category] == element
             indexes_nonzero = np.nonzero(np.array(connected))[0]
             
             for idx_nz in indexes_nonzero:
                 coo_matrix_fill[key_dict[element], idx_nz + start] = 1
                 coo_matrix_fill[idx_nz + start, key_dict[element]] = 1
        
        return coo_matrix_fill

    def gen_key_adj_grap(authors, total_nodes, school, timeframe, studied_df, types, key_dicts, start):
        kg_coo = dok_matrix((total_nodes, total_nodes), dtype=np.int8)
        generate_connects(kg_coo, studied_df, 'AUTHOR', key_dicts, start)
        generate_connects(kg_coo, studied_df, 'SCHOOL', key_dicts, start)
        generate_connects(kg_coo, studied_df, 'TIMEFRAME', key_dicts, start)
        generate_connects(kg_coo, studied_df, 'TYPE', key_dicts, start)

        return key_dicts, kg_coo

    train_df = pd.read_csv(semart_path + r'/semart_train.csv', sep='\t', encoding='latin1')
    val_df = pd.read_csv(semart_path + r'/semart_val.csv', sep='\t', encoding='latin1')
    test_df = pd.read_csv(semart_path + r'/semart_test.csv', sep='\t',  encoding='latin1')
    
    authors = np.unique(train_df['AUTHOR'])
    school = np.unique(train_df['SCHOOL'])
    timeframe = np.unique(train_df['TIMEFRAME'])
    types = np.unique(train_df['TYPE'])
    
    extra_nodes = len(authors) + len(school) + len(timeframe) + len(types)

    all_categories = np.concatenate((authors, school, timeframe, types))
    key_dicts = {x: ix + train_df.shape[0] for ix, x in enumerate(all_categories)}

    key_dicts, kg_coo = gen_key_adj_grap(authors, train_df.shape[0] + extra_nodes, school, timeframe, train_df, types, key_dicts, start=0)
    val_key_dicts, val_kg_coo = gen_key_adj_grap(authors, kg_coo.shape[0] + val_df.shape[0], school, timeframe, val_df, types, key_dicts, start=train_df.shape[0] + extra_nodes)
    test_key_dicts, test_kg_coo = gen_key_adj_grap(authors, val_kg_coo.shape[0] + test_df.shape[0], school, timeframe, test_df, types, key_dicts, train_df.shape[0] + val_df.shape[0] + extra_nodes)

    return [kg_coo, key_dicts], [val_kg_coo, val_key_dicts], [test_kg_coo, test_key_dicts]


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
    
    with open(res_file, 'w', encoding='latin1') as f:
        for key, elem in key_dict.items():
            f.write(str(key.replace(' ', '_')) + ' ' + str(elem) + '\n')
    
if __name__ == '__main__':
    [kg_semart, keys_semart], [val_kg_coo, val_key_dicts], [test_kg_coo, test_key_dicts] = generate_kg_semart()
    
    save_edges(kg_semart)
    save_keys(keys_semart)

    save_edges(val_kg_coo, 'kg_semart_val.csv')
    save_keys(val_key_dicts, 'kg_keys_val.csv')

    save_edges(test_kg_coo, 'kg_semart_test.csv')
    save_keys(test_key_dicts, 'kg_keys_test.csv')
