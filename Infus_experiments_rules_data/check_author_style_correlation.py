'''

Experiments to reproduce one table from the infus paper.

'''

import os
import numpy as np
import pandas as pd
from unidecode import unidecode

# Load the metadata from wikiart
wikiart_path = '/home/javierfumanal/Downloads/wikiart/'
semart_path = '/home/javierfumanal/Documents/GitHub/SemArt/'
artist_number_dictionary_file = wikiart_path + 'artist_class.txt'
artist_number_dictionary = {}
number_author_dictionary = {}
with open(artist_number_dictionary_file, 'r') as f:
    for line in f:
        number, artist = line.split()
        artist_number_dictionary[artist.lower()] = int(number)
        number_author_dictionary[int(number)] = artist

wikiart_artists = pd.read_csv(wikiart_path + 'artist_train.csv', index_col=0)
wikiart_styles = pd.read_csv(wikiart_path + 'style_train.csv', index_col=0)

# Get only the paintings that have both a style and an author
wikiart_styles = wikiart_styles.loc[wikiart_styles.index.isin(wikiart_artists.index)]
wikiart_artists = wikiart_artists.loc[wikiart_artists.index.isin(wikiart_styles.index)]

lkeys = [key.lower() for key in  artist_number_dictionary.keys()]

# Load the metadata from semart
semart_train = pd.read_csv(semart_path + 'semart_train.csv', encoding='cp1252',  delimiter='\t')
sem2dict = lambda a : '_'.join(unidecode(a).lower().split(',')[::-1]).strip().replace(' ', '_')

semart_artists = np.unique(semart_train['AUTHOR'].apply(sem2dict))
semart_train['AUTHOR'] = semart_train['AUTHOR'].apply(sem2dict)

# Check the common ones
common_artists = np.intersect1d(semart_artists, lkeys)
print('Common artists: ', len(common_artists))

rows_with_common_paintings = semart_train[semart_train['AUTHOR'].isin(common_artists)]['IMAGE_FILE']

# Load the explainable features from the Infus paper
infus_features = pd.read_csv('./Infus_experiments_rules_data/explainable_features.csv', index_col=0)
infus_features = infus_features.loc[rows_with_common_paintings]
semart_style_features = infus_features.iloc[:, :-4]
semart_style_preds = semart_style_features.idxmax(axis=1)

# Check the style of the common authors
res_wikiart = pd.DataFrame(np.zeros((len(common_artists), len(semart_style_features.columns))), columns=semart_style_features.columns, index=common_artists)
res_semart = pd.DataFrame(np.zeros((len(common_artists), len(semart_style_features.columns))), columns=semart_style_features.columns, index=common_artists)

for ax, author in enumerate(common_artists):
    # Check the styles in the wikiart dataset
    artist_id = artist_number_dictionary[author]
    wikiart_styles_of_author = list(wikiart_styles.loc[(wikiart_artists.values == artist_id).flatten()].index)

    styles = [x.split('/')[0] for x in wikiart_styles_of_author]
    style_wikiart_counts = {}
    for style in styles:
        if style in style_wikiart_counts:
            style_wikiart_counts[style] += 1
        else:
            style_wikiart_counts[style] = 1
    
    # Check the styles in the semart dataset
    semart_paintings_artist = semart_train.loc[semart_train['AUTHOR'] == author]['IMAGE_FILE']
    semart_style_preds.loc[semart_paintings_artist].value_counts()
    
    # Compare the styles predicted
    style_semart_counts = {}
    for style in semart_style_preds.loc[semart_paintings_artist].values:
        if style in style_semart_counts:
            style_semart_counts[style] += 1
        else:
            style_semart_counts[style] = 1

    print(author)
    print('Wikiart: ', style_wikiart_counts)
    print('Semart: ', style_semart_counts)

    res_wikiart.loc[author] = style_wikiart_counts
    res_semart.loc[author] = style_semart_counts

res_wikiart.fillna(0, inplace=True)
res_semart.fillna(0, inplace=True)

# res_wikiart[:] = res_wikiart.values / res_wikiart.sum(axis=1).values[:, None]
# res_semart[:] = res_semart.values / res_semart.sum(axis=1).values[:, None]

#Compute the number of non-zeros elements in each row for both datasets
non_zeros_wikiart = np.count_nonzero(res_wikiart.values, axis=1)
non_zeros_semart = np.count_nonzero(res_semart.values, axis=1)
divisor = np.maximum(non_zeros_wikiart, non_zeros_semart)

abs_error = np.abs(res_wikiart - res_semart).sum(axis=1) / divisor
print('Done')