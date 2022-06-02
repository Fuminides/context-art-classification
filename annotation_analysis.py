import os
import pickle
import itertools

import pandas as pd
import numpy as np

import params
import utils

from Data import symbol_graph as sg

class dummyPlug: #EVA 01: YURI, DONT DO THIS TO ME!
    def __init__(self):
        pass

try:
    args_dict
except NameError:
    args_dict = dummyPlug()
    args_dict.mode = 'train'
    args_dict.dir_dataset = r'G:\Mi unidad\Code\SemArt'
    args_dict.csvtrain = args_dict.dir_dataset + '\semart_train.csv'

## FUNCTIONS RELATED TO THE MYTH GRAPHS
def edges2adjacency_df(edges_df,symmetry=False):
    '''
    Returns the adjacency version of an edges df.
    '''
    unicos = set(edges_df['Source']).union(set(edges_df['Target']))

    res = pd.DataFrame(np.zeros((len(unicos), len(unicos))))
    res.columns = unicos
    res.index = unicos

    for ix, elem_row in enumerate(edges_df.iterrows()):
        try:
            source, target, weight = elem_row[1]
        except ValueError:
            source, target = elem_row[1]
            weight = 1
            
        res[source][target] = weight
        if symmetry:
            res[target][source] = weight

    return res

def load_edges_edda():
    aux = pd.read_csv('Data/edda_edges_df.csv', index_col=0)
    aux['Source'] = aux['Source'].apply(str.lower)
    aux['Target'] = aux['Target'].apply(str.lower)

    return aux

def load_edges_celt():
    aux = pd.read_csv('Data/celt_edges_df.csv', index_col=0)
    aux['Source'] = aux['Source'].apply(str.lower)
    aux['Target'] = aux['Target'].apply(str.lower)

    return aux

def load_edges_odyssey():
    aux = pd.read_csv('Data/odyssey_edges_df.csv', index_col=0)
    aux['Source'] = aux['Source'].apply(str.lower)
    aux['Target'] = aux['Target'].apply(str.lower)

    return aux

def load_edges_iliad():
    aux = pd.read_csv('Data/iliad_edges_df.csv', index_col=0)
    aux['Source'] = aux['Source'].apply(str.lower)
    aux['Target'] = aux['Target'].apply(str.lower)

    return aux

def load_edges_greek():
    aux = pd.read_csv('Data/greek_edges_df.csv', index_col=0)
    aux['Source'] = aux['Source'].apply(str.lower)
    aux['Target'] = aux['Target'].apply(str.lower)

    return aux

def load_edges_myth():
    edda = load_edges_edda()
    celt = load_edges_celt()
    greek = load_edges_greek()

    myth_edges = pd.concat([edda, celt, greek])

    return myth_edges.drop_duplicates()

def load_edda():
    return edges2adjacency_df(pd.read_csv('Data/edda_edges_df.csv', index_col=0), True)

def load_celt():
    return  edges2adjacency_df(pd.read_csv('Data/celt_edges_df.csv', index_col=0), True)

def load_greek():
    return  edges2adjacency_df(pd.read_csv('Data/greek_edges_df.csv', index_col=0), True)

def load_odyssey():
    return  edges2adjacency_df(pd.read_csv('Data/odyssey_edges_df.csv', index_col=0), True)

def load_iliad():
    return  edges2adjacency_df(pd.read_csv('Data/iliad_edges_df.csv', index_col=0), True)

def load_myth():
    return  edges2adjacency_df(pd.read_csv('Data/myth_all.csv', index_col=0), True)

def filter_df(wanted_names, tale_df):
    actual_names = tale_df.index
    filtered = [name for name in actual_names if name in wanted_names]
    tale_df = tale_df.copy()
    tale_df = tale_df.loc[filtered, :]
    tale_df = tale_df.T.loc[filtered, :]

    return tale_df.T

def filter_edges_df(wanted_names, edges_df):
    filtered_edge_df = pd.DataFrame(None, columns=['Source', 'Target'])
    for _, row in edges_df.iterrows():
        source = row['Source']
        target = row['Target']

        if source in wanted_names and target in wanted_names:
            filtered_edge_df = pd.concat([filtered_edge_df, pd.DataFrame.from_dict({'Source': [source], 'Target': [target]})])
        
    
    return filtered_edge_df

def filter_dual_edges_df(edges_df1, edges_df2):
    wanted_names1 = set(pd.concat([edges_df1['Source'], edges_df1['Target']]))
    wanted_names2 = set(pd.concat([edges_df2['Source'], edges_df2['Target']]))
    
    new_df1 = filter_edges_df(wanted_names1, edges_df2)
    new_df2 = filter_edges_df(wanted_names2, edges_df1)

    return new_df1, new_df2

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
    global args_dict
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
    names = df['TITLE']
    descriptions = df['DESCRIPTION'] # Load the contextual annotations

    dictionary_painting_symbol = np.zeros((df.shape[0], len(symbol_canon_list)))
    for ix, description in enumerate(descriptions):
        
        symbols_painting = np.zeros((len(symbol_canon_list),))

        for jx, symbol in enumerate(symbol_canon_list):
            symbol = symbol.lower()
            description = description.lower()

            if symbol in description.split():
                symbols_painting[jx] = 1

        dictionary_painting_symbol[ix, :] = symbols_painting
    
    return dictionary_painting_symbol.astype(np.bool), names, symbol_canon_list

def symbol_connectivity():
    from Data import symbol_graph

    trial = symbol_graph.generate_adjacency_df_symbol(symmetry=True)

    return trial

def symbol_report(symbol_context, symbol_names=None, painting_names=None):

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
    symbol_histogram = symbol_context.sum(axis=0)
    res['symbol_histogram'] = symbol_histogram

    #Most common symbols
    symbol_histogram = np.argsort(symbol_context.sum(axis=0))
    if symbol_names is None:
        res['sorted_symbols'] = symbol_histogram
    else:
        res['sorted_symbols'] = symbol_names[symbol_histogram]


    return res

def paint_gallery(painting_list, symbol_mat, symbols_names):
    max_width = 3
    rows = (len(painting_list) % max_width) + 1
    i = 1
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(30,30))

    for painting in painting_list:
        fig.add_subplot(2, max_width, i) # two rows, one column, first plot
        vis_painting_symbols(painting, symbol_mat, symbols_names, symbols_mode=False)
        i+=1    

def most_import_connections(symbol_painting_graphs, k=10):
   s_names = symbol_painting_graphs.index 
   def get_indices_of_k_smallest(arr, k):
    idx = np.argpartition(arr.ravel(), k)
    return tuple(np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))])

   aux = np.tril(symbol_painting_graphs, k=-1)
   idx_array = get_indices_of_k_smallest(aux*-1, k=k)

   return [(s_names[a], s_names[b]) for a,b in zip(*idx_array)]

def vis_painting_symbols(painting_arg, symbol_mat, symbols_names, symbols_mode=True):

        
    def write_symbols(symbol_list):
        start_x = 0.8
        start_y = 0.8
        x_add = 0.15
        y_add = -0.1
        for symbol in symbol_list:
            plt.text(start_x, start_y, symbol, fontsize=14, transform=plt.gcf().transFigure)

            start_y += y_add

            if  start_y < 0.1:
                start_x += x_add
                start_y = 0.8

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    textfile = os.path.join(args_dict.dir_dataset, args_dict.csvtrain)
    df = pd.read_csv(textfile, delimiter='\t', encoding='Cp1252')

    if isinstance(painting_arg, str):
        trial = df['TITLE'] == painting_arg
        if sum(trial) == 0:
            trial = df['IMAGE_FILE'] == painting_arg
        
        painting_arg = np.argmax(trial)
    
    
    my_image = mpimg.imread(args_dict.dir_dataset + '/Images/' + df['IMAGE_FILE'].iloc[painting_arg])

    name = df['TITLE'].iloc[painting_arg]
    symbols = symbol_mat[painting_arg, :]
    symbols_names_painting = symbols_names[symbols.astype(np.bool)]


    #plt.figure()
    plt.title(name)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    if symbols_mode:
        write_symbols(symbols_names_painting)

    plt.imshow(my_image)


def load_semart_annotations_titles(args_dict):
    '''
    Loads for each painting the title + annotation
    '''
    from Data.symbol_graph import CIRLO_DICT

    dict_mix = pd.read_csv(CIRLO_DICT)
    df = pd.read_csv(os.path.join(args_dict.dir_dataset, args_dict.csvtrain), delimiter='\t', encoding='Cp1252')

    terms = df['TITLE']  
    definitions = dict_mix['ANNOTATION']

    return zip(terms, definitions)  

def most_connected_symbols(symbol_context, symbols_names, k=10):
    idx = (symbol_context.to_numpy().sum(axis=0)*-1).argsort()[0:k]
    counts = symbol_context.sum(axis=0)[idx]
    return list(zip(symbols_names[idx], counts))

def most_connected_symbols_theme(symbol_context, symbols_names, k=10):
    idx = (symbol_context.to_numpy().sum(axis=0)*-1).argsort()[0:k]
    counts = symbol_context.sum(axis=0)[idx]
    return list(zip(symbols_names[idx], counts))

def more_repeated_symbols_theme(symbol_context, symbols_names, theme, df, k=10):
    chosen_paintings = df['THEME'] == theme
    return list(symbols_names[chosen_paintings[chosen_paintings, :].sum(axis=0).argsort()[::-1][0:k]].values())

def semart_gen_symbol_graph(symbol_context):
    try:
        symbol_context = symbol_context.toarray()
    except AttributeError:
        pass
    res = np.zeros((symbol_context.shape[1], symbol_context.shape[1]))
    for painting in range(symbol_context.shape[0]):            
        painting_symbols = symbol_context[painting, :].squeeze()
        existing_symbols = [i for i, e in enumerate(painting_symbols) if e > 0]
        iter_list_symbols = list(itertools.product(existing_symbols, existing_symbols))
        for s1, s2 in iter_list_symbols:
            res[s1, s2] += 1
            res[s2, s1] += 1

    return res / 2

def semart_gen_painting_graph(symbol_context):
    res = np.zeros((symbol_context.shape[0], symbol_context.shape[0]))
    for symbol in range(symbol_context.shape[1]):
        painting_symbols = symbol_context[:, symbol].toarray().squeeze()
        existing_symbols = [i for i, e in enumerate(painting_symbols) if e != 0]
        iter_list_symbols = list(itertools.product(existing_symbols, existing_symbols))
        for s1, s2 in iter_list_symbols:
            res[s1, s2] += 1
            res[s2, s1] += 1

    return res
    
#### COMPARISON  METHODS BTWEEN CIRLOT AND SEMART

def symbol_existence(symbol_context):
    return np.mean(symbol_context.sum(axis=0)>0)

def pair_graph_load():
    symbol_context, paintings_names, symbols_names = __load_semart_proxy(mode='train')
    cirlot_semart_reduced = sg.generate_adjacency_symbol_sparse_reduced(symbols_names, symmetry=True)

    return utils.graph_similarity(cirlot_semart_reduced, symbol_context)



if __name__ == '__main__':
    symbol_context, paintings_names, symbols_names = __load_semart_proxy(mode='train')
    #res = symbol_report(symbol_context, symbol_names=symbols_names, painting_names=paintings_names)
    res = semart_gen_symbol_graph(symbol_context)
    print(res)
    print('hOLA')