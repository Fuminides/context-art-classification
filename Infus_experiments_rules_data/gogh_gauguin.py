import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/javierfumanal/Documents/GitHub/FuzzyT2Tbox/ex_fuzzy/')
sys.path.append('/home/fcojavier.fernandez/Github/FuzzyT2Tbox/ex_fuzzy/')
sys.path.append('C:/Users/javier.fumanal/Documents/GitHub/FuzzyT2Tbox/ex_fuzzy/')

import ex_fuzzy.fuzzy_sets as fs
import ex_fuzzy.evolutionary_fit as GA
import ex_fuzzy.utils as utils
import ex_fuzzy.eval_tools as eval_tools
import ex_fuzzy.rules as rules
import ex_fuzzy.eval_rules as evr
from sklearn.model_selection import train_test_split


def load_explainable_features(path='/home/javierfumanal/Documents/GitHub/FuzzyT2Tbox/Demos/occupancy_data/explainable_features.csv', sample_ratio=0.1):
    try:
        data = pd.read_csv(path, index_col=0).sample(frac=sample_ratio, random_state=33)
    except FileNotFoundError:
        try:
            data = pd.read_csv('/home/fcojavier.fernandez/Github/FuzzyT2Tbox/Demos/occupancy_data/explainable_features.csv', index_col=0).sample(frac=sample_ratio, random_state=33)
        except FileNotFoundError:
            data = pd.read_csv('C:/Users/javier.fumanal/Documents/GitHub/FuzzyT2Tbox/Demos/occupancy_data/explainable_features.csv', index_col=0).sample(frac=sample_ratio, random_state=33)
    
    data.fillna(0, inplace=True)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=33, stratify=y) #Como que 33?

    return X, y, X_train, X_test, y_train, y_test

# Load SemArt dataset
semart_path = 'C:/Users/javier.fumanal/Documents/GitHub/SemArt/'
semart_train = pd.read_csv(semart_path + 'semart_train.csv', encoding='cp1252', sep='\t')

van_gogh = semart_train[semart_train['AUTHOR'] == 'GOGH, Vincent van']
paul = semart_train[semart_train['AUTHOR'] == 'GAUGUIN, Paul']

load_feats = load_explainable_features(sample_ratio=1.0)

try:
    n_gen = int(sys.argv[1])
    pop_size = int(sys.argv[2])
    nRules = int(sys.argv[3])
    nAnts = int(sys.argv[4])
    runner = int(sys.argv[5])
except:
    print('Using default parameters')
    n_gen = 1000
    pop_size = 30
    nRules = 15
    nAnts = 4
    runner = 1


painter = 'GOGH'
fz_type_studied = fs.FUZZY_SETS.t1
checkpoints = 0

X, y, X_train, X_test, y_train, y_test = load_explainable_features(sample_ratio=1.0)
X_van_gogh = X.loc[van_gogh['IMAGE_FILE']]
X_paul = X.loc[paul['IMAGE_FILE']]
y_artists = np.zeros(X_van_gogh.shape[0] + X_paul.shape[0])
y_artists[:X_van_gogh.shape[0]] = 1
X_artists = pd.concat([X_van_gogh, X_paul])

X_artists_train, X_artists_test, y_artists_train, y_artists_test = train_test_split(X_artists, y_artists, test_size=0.10, random_state=33, stratify=y_artists) #Como que 33?

precomputed_partitions = utils.construct_partitions(X, fz_type_studied)
#precomputed_partitions=None
min_bounds = np.min(X, axis=0).values
max_bounds = np.max(X, axis=0).values
domain = [min_bounds, max_bounds]

print('Training fuzzy classifier:' , nRules, 'rules, ', nAnts, 'ants, ', n_gen, 'generations, ', pop_size, 'population size')
fl_classifier = GA.FuzzyRulesClassifier(nRules=nRules, nAnts=nAnts, 
    linguistic_variables=precomputed_partitions, n_linguist_variables=3, 
    fuzzy_type=fz_type_studied, verbose=True, tolerance=0.0, domain=domain, runner=runner)
# fl_classifier.customized_loss(new_loss)
fl_classifier.fit(X_artists_train, y_artists_train, n_gen=n_gen, pop_size=pop_size, checkpoints=checkpoints)

str_rules = eval_tools.eval_fuzzy_model(fl_classifier, X_artists_train, y_artists_train, X_artists_test, y_artists_test, 
                        plot_rules=False, print_rules=True, plot_partitions=False, return_rules=True)

# Check for other rule bases in the folder
import os
res_folder = 'rules_art_features'
if not os.path.exists(res_folder):
    os.makedirs(res_folder)

files = os.listdir('rules_art_features')
# We save the rules in a file
with open('rules_art_features/rules_feature' + str(painter) + '.txt', 'w') as f:
    f.write(str_rules)

print('Done')




