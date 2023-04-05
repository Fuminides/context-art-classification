'''
This script is used to run the evolutionary algorithm to find the best fuzzy rule base for the explainable features experiment
for the SemArt dataset.

The script takes 5 arguments:
    - n_gen: Number of generations
    - pop_size: Population size
    - nRules: Number of rules
    - nAnts: Number of antecedents per rule
    - n_threads: Number of threads to use

Example:
    python art_demo.py 10 30 10 3 2

This will run the algorithm for 10 generations, with a population size of 30, 10 rules, 3 antecedents per rule, and 2 threads.

The script will save the rules in a file called rules_#.txt, where # is the number of the file. If there are no files in the
folder, the number will be 0. If there are files, the number will be the last number + 1.

The script will also print the rules in the console.

author: Javier Fumanal Idocin

'''
import pandas as pd
import numpy as np

import sys
sys.path.append('/home/javierfumanal/Documents/GitHub/FuzzyT2Tbox/ex_fuzzy/')
sys.path.append('/home/fcojavier.fernandez/Github/FuzzyT2Tbox/ex_fuzzy/')
import ex_fuzzy.fuzzy_sets as fs
import ex_fuzzy.evolutionary_fit as GA
import ex_fuzzy.utils as utils
import ex_fuzzy.eval_tools as eval_tools
from sklearn.model_selection import train_test_split

from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization

def load_explainable_features(path='/home/javierfumanal/Documents/GitHub/FuzzyT2Tbox/Demos/occupancy_data/explainable_features.csv', sample_ratio=0.1):
    try:
        data = pd.read_csv(path, index_col=0).sample(frac=sample_ratio, random_state=33)
    except FileNotFoundError:
        data = pd.read_csv('/home/fcojavier.fernandez/Github/FuzzyT2Tbox/Demos/occupancy_data/explainable_features.csv', index_col=0).sample(frac=sample_ratio, random_state=33)
        
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33) #Como que 33?

    return X, y, X_train, X_test, y_train, y_test


try:
    n_gen = int(sys.argv[1])
    pop_size = int(sys.argv[2])
    nRules = int(sys.argv[3])
    nAnts = int(sys.argv[4])
    n_threads = int(sys.argv[5])
except:
    n_gen = 100
    pop_size = 30
    nRules = 20
    nAnts = 3
    n_threads = 1

if n_threads > 1:
    pool = ThreadPool(n_threads)
    runner = StarmapParallelization(pool.starmap)
else:
    runner = None

fz_type_studied = fs.FUZZY_SETS.t1

X, y, X_train, X_test, y_train, y_test = load_explainable_features(sample_ratio=0.05)
precomputed_partitions = utils.construct_partitions(X, fz_type_studied)
min_bounds = np.min(X, axis=0).values
max_bounds = np.max(X, axis=0).values
domain = [min_bounds, max_bounds]

print('Training fuzzy classifier:' , nRules, 'rules, ', nAnts, 'ants, ', n_gen, 'generations, ', pop_size, 'population size')
fl_classifier = GA.FuzzyRulesClassifier(nRules=nRules, nAnts=nAnts, 
    linguistic_variables=precomputed_partitions, n_linguist_variables=3, 
    fuzzy_type=fz_type_studied, verbose=True, tolerance=0.0, domain=domain, runner=runner)

fl_classifier.fit(X_train, y_train, n_gen=n_gen, pop_size=pop_size, checkpoints=0)

str_rules = eval_tools.eval_fuzzy_model(fl_classifier, X_train, y_train, X_test, y_test, 
                        plot_rules=False, print_rules=True, plot_partitions=False, return_rules=True)

# Check for other rule bases in the folder
import os
files = os.listdir('rules_art_features')
files = [f for f in files if 'rules_' in f]
if len(files) > 0:
    files.sort()
    last_file = files[-1]
    last_num = int(last_file.split('_')[1].split('.')[0])
    new_num = last_num + 1
else:
    new_num = 0

# We save the rules in a file
with open('rules_art_features/rules_candidates_' + str(new_num) + '.txt', 'w') as f:
    f.write(str_rules)

print('Done')
