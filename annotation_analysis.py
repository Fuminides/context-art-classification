import os
import pickle
import itertools

import pandas as pd
import numpy as np

import params
import utils

from Data import symbol_graph as sg

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import hashlib

class dummyPlug: #EVA 01: YURI, DONT DO THIS TO ME!
    def __init__(self):
        pass


## FUNCTIONS RELATED TO THE ARTEMIS DATASET
def load_artemis_df():
    return pd.read_csv('Data/artemis_dataset_release_v0.csv')

def check_symbols_feelings():
    if os.path.exists('Data/feelings_symbols.csv'):
        return pd.read_csv('Data/feelings_symbols.csv')
    else:
        artemis_df = load_artemis_df()
        emotions = artemis_df['emotion'].unique()

        symbols_names = load_terms()
        res = pd.DataFrame(np.zeros((len(symbols_names), len(emotions))), index=symbols_names, columns=emotions)

        for ix, row in artemis_df.iterrows():
            text, feeling = row['utterance'], row['emotion']
            words = text.split()
            words = [word.lower() for word in words]

            for symbol in symbols_names:
                if symbol.lower() in words:
                    res.loc[symbol, feeling] += 1
    
    return res

def most_striking_symbols(k=10):
    '''
    Returns the symbols with the highest number of occurrences in the myth graph.
    '''
    return list(zip(load_artemis_df().sum(axis=1).sort_values(ascending=False).index[:k],
                load_artemis_df().sum(axis=1).sort_values(ascending=False).values[:k]))

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
    
    args_dict = dummyPlug()
    args_dict.mode = 'train'
    args_dict.dir_dataset = r'C:/Users/jf22881/Documents/SemArt'
    args_dict.csvtrain =  'semart_train.csv'
    return load_semart_symbols(args_dict, dataset='train', strict_names=True)

def load_semart_symbols(args_dict, dataset, strict_names=False):
    # Load data
    if dataset == 'train':
        textfile = os.path.join(args_dict.dir_dataset, args_dict.csvtrain)
    elif dataset == 'val':
        textfile = os.path.join(args_dict.dir_dataset, args_dict.csvval)
    elif dataset == 'test':
        textfile = os.path.join(args_dict.dir_dataset, args_dict.csvtest)
    
    try:
        if strict_names:
            symbol_canon_list = args_dict.canon_list
        else:
            symbol_canon_list = list(set([each_string.lower() for each_string in load_terms()]) & set(args_dict.canon_list)) 
    except:
        symbol_canon_list = load_terms()
    print('Loading file... ' + str(textfile))

    df = pd.read_csv(textfile, delimiter='\t', encoding='Cp1252')
    names = df['TITLE']
    descriptions = df['TITLE'] + ' ' + df['DESCRIPTION'] # Load the contextual annotations

    hash_cached = str(hashlib.sha224(str.encode(str(df))).hexdigest())

    
    if os.path.exists('cache/' + hash_cached):
        print('Symbols cached... ')
        dictionary_painting_symbol = pd.read_csv('cache/' + hash_cached, index_col=0).values
    else:
        print('Computing Symbols matrix... ')
        dictionary_painting_symbol = np.zeros((df.shape[0], len(symbol_canon_list)))
        for ix, description in enumerate(descriptions):
        
            symbols_painting = np.zeros((len(symbol_canon_list),))

            for jx, symbol in enumerate(symbol_canon_list):
                symbol = symbol.lower()
                description = description.lower()

                if symbol in description.split():
                    symbols_painting[jx] = 1

            dictionary_painting_symbol[ix, :] = symbols_painting

        if not strict_names:
            useful_symbols = np.sum(dictionary_painting_symbol, axis=0) > 0
            dictionary_painting_symbol = dictionary_painting_symbol[:, useful_symbols]
            symbol_canon_list = [x for ix, x in enumerate(symbol_canon_list) if useful_symbols[ix]]

        pd.DataFrame(dictionary_painting_symbol).to_csv('cache/' + hash_cached)

    return dictionary_painting_symbol.astype(np.bool), names, symbol_canon_list

def symbol_connectivity():
    from Data import symbol_graph

    trial = symbol_graph.generate_adjacency_df_symbol(symmetry=True)

    return trial

def most_import_connections(symbol_painting_graphs, k=10):
   s_names = symbol_painting_graphs.index 
   def get_indices_of_k_smallest(arr, k):
    idx = np.argpartition(arr.ravel(), k)
    return tuple(np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))])

   aux = np.tril(symbol_painting_graphs, k=-1)
   idx_array = get_indices_of_k_smallest(aux*-1, k=k)

   return [(s_names[a], s_names[b]) for a,b in zip(*idx_array)]

def load_semart_annotations_titles(args_dict):
    '''
    Loads for each painting the title + annotation
    '''
    from Data.symbol_graph import CIRLO_DICT

    dict_mix = pd.read_csv(CIRLO_DICT)
    df = pd.read_csv(os.path.join(args_dict.dir_dataset, args_dict.csvtrain), delimiter='\t', encoding='Cp1252')

    terms = df['TITLE']  
    definitions = df['TITLE']   + ' ' + dict_mix['ANNOTATION']

    return zip(terms, definitions)  

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
   
#### COMPARISON  METHODS BTWEEN CIRLOT AND SEMART
def pair_graph_load():
    symbol_context, paintings_names, symbols_names = __load_semart_proxy(mode='train')
    cirlot_semart_reduced = sg.generate_adjacency_symbol_sparse_reduced(symbols_names, symmetry=True)

    return utils.graph_similarity(cirlot_semart_reduced, symbol_context)

class Gallery:

    def __init__(self,symbols_names, paintings_names, symbol_context, path):
        self.symbols_names = symbols_names
        self.painting_names = paintings_names
        self.symbol_context = symbol_context
        self.path = path

        textfile = os.path.join(self.path, 'semart_train.csv')
        self.df = pd.read_csv(textfile, delimiter='\t', encoding='Cp1252')
        

    def symbol_existence(self):
        return np.mean(self.symbol_context.sum(axis=0)>0)
    
    def semart_gen_painting_graph(self):
        res = np.zeros((self.symbol_context.shape[0], self.symbol_context.shape[0]))
        for symbol in range(self.symbol_context.shape[1]):
            painting_symbols = symbol_context[:, symbol].toarray().squeeze()
            existing_symbols = [i for i, e in enumerate(painting_symbols) if e != 0]
            iter_list_symbols = list(itertools.product(existing_symbols, existing_symbols))
            for s1, s2 in iter_list_symbols:
                res[s1, s2] += 1
                res[s2, s1] += 1

        return res

    def vis_painting_symbols(self, painting_arg, symbols_mode=True):

        def write_symbols(symbols_names_list):
            start_x = 1
            start_y = 0.8
            x_add = 0.30
            y_add = -0.1
            for symbol in symbols_names_list:
                plt.text(start_x, start_y, symbol, fontsize=14, transform=plt.gcf().transFigure)

                start_y += y_add

                if  start_y < 0.1:
                    start_x += x_add
                    start_y = 0.8
      
        if isinstance(painting_arg, str):
            trial = self.df['TITLE'] == painting_arg
            if sum(trial) == 0:
                trial = self.df['IMAGE_FILE'] == painting_arg
            
            painting_arg = np.argmax(trial)
        
        my_image = mpimg.imread(self.path + '/Images/' + self.df['IMAGE_FILE'].iloc[painting_arg])

        name = self.df['TITLE'].iloc[painting_arg]
        symbols = self.symbol_context[painting_arg, :]
        print(len(self.symbols_names), len(symbols))
        symbols_names_painting = [x for ix, x in enumerate(self.symbols_names) if symbols.astype(np.bool)[ix]] # self.symbols_names[symbols.astype(np.bool)]


        #plt.figure()
        plt.title(name)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        if symbols_mode:
            write_symbols(symbols_names_painting)

        plt.imshow(my_image)

    def symbol_report(self):

        res = {}
        #Number of paintings with at least one symbol
        symbol_presence = np.mean(self.symbol_context.sum(axis=1)>0)
        res['symbol_paintings'] = symbol_presence

        #Number of paintings with at more than one symbol
        symbol_abundance = np.mean(self.symbol_context.sum(axis=1)>1)
        res['useful_paintings'] = symbol_abundance

        #Histogram of number of symbols per painting
        symbol_histogram = self.symbol_context.sum(axis=1)
        res['painting_histogram'] = symbol_histogram

        #Paintings sortered with more symbols
        symbol_histogram = np.argsort(self.symbol_context.sum(axis=1))
        
        
        res['important_paintings'] = self.painting_names[symbol_histogram]

        #Symbols that appear at least one time
        symbol_histogram = np.mean(self.symbol_context.sum(axis=0) > 0)
        res['useful_symbols'] = symbol_histogram

        #Histogram of number of symbols
        symbol_histogram = self.symbol_context.sum(axis=0)
        res['symbol_histogram'] = symbol_histogram

        #Most common symbols
        symbol_histogram = np.argsort(self.symbol_context.sum(axis=0))
        
        res['sorted_symbols'] = self.symbols_names[symbol_histogram]


        return res

    def paint_gallery(self, painting_list):
        max_width = 3
        rows = (len(painting_list) % max_width) + 1
        i = 1
        fig = plt.figure(figsize=(30,30))

        for painting in painting_list:
            fig.add_subplot(2, max_width, i) # two rows, one column, first plot
            self.vis_painting_symbols(painting, symbols_mode=False)
            i+=1    
        
    
    def most_repeated_symbols_theme(self, theme, k=10):
        chosen_paintings = self.df['TYPE'] == theme
        return list(self.symbols_names[self.symbol_context[chosen_paintings, :].sum(axis=0).argsort()[::-1][0:k]])
    
    def most_repeated_symbols(self, k=10):
        symbols_args = self.symbol_context.sum(axis=0).argsort()[::-1][0:k]
        return [list(self.symbols_names)[ix] for ix in symbols_args]


    def most_connected_symbols(self, k=10):
        idx = (self.symbol_context.sum(axis=0)*-1).argsort()[0:k]
        counts = self.symbol_context.sum(axis=0)[idx]
        return list(zip(self.symbols_names[idx], counts))    
    
    def n_symbolic_paintings(self, k=None):
        aux = np.argsort(self.symbol_context.sum(axis=1))[::-1]

        if aux is not None:
            aux = aux[0:k]
        
        return aux

    def density_symbolic_paintings(self, k=None):
        aux = np.argsort(self.symbol_context.mean(axis=1))[::-1]

        if aux is not None:
            aux = aux[0:k]
        
        return aux
        



if __name__ == '__main__':
    symbol_context, paintings_names, symbols_names = __load_semart_proxy(mode='train')
    semart_Gallery = Gallery(symbols_names, paintings_names, symbol_context, '../SemArt/')

    #res = symbol_report(symbol_context, symbol_names=symbols_names, painting_names=paintings_names)
    res = semart_gen_symbol_graph(symbol_context)
    print(res)
    print('hOLA')