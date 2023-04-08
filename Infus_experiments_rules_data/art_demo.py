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

The script will save the rules in a file called rules_#.txt, where # is the featuer number.

The script will also print the rules in the console.

author: Javier Fumanal Idocin

'''
import pandas as pd
import numpy as np

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


# Matthew Correlation Coefficient
from sklearn.metrics import matthews_corrcoef
# Supress warnings
import warnings
warnings.filterwarnings("ignore")

def new_loss(ruleBase: rules.RuleBase, X:np.array, y:np.array, tolerance:float):
        '''
        Fitness function for the optimization problem. 
        The fitness function is the accuracy of the rule base, weighted by the size of the rule base.
        It creates a balanced random partition of the data to evaluate the rule base.

        :param ruleBase: RuleBase object
        :param X: array of train samples. X shape = (n_samples, n_features)
        :param y: array of train labels. y shape = (n_samples,)
        :param tolerance: float. Tolerance for the size evaluation.
        :return: float. Fitness value.
        '''
        # Susbsample X and y to have balanced classes
        positive_index = np.where(y)[0]
        negative_index = np.where(y == 0)[0]
        negative_samples_index = np.random.choice(negative_index, len(positive_index), replace=False)
        total_index = np.concatenate((positive_index, negative_samples_index))
        #Shuffle the index
        np.random.shuffle(total_index)
        X_balanced = X[total_index]
        y_balanced = y[total_index]
        
        ev_object = evr.evalRuleBase(ruleBase, X_balanced, y_balanced)
        ev_object.add_rule_weights()

        score_acc = ev_object.classification_eval()
        score_size = ev_object.size_eval(tolerance)

        alpha = 0.99
        beta = 1 - alpha

        score = score_acc * alpha + score_size * beta

        return score


def load_explainable_features(path='/home/javierfumanal/Documents/GitHub/FuzzyT2Tbox/Demos/occupancy_data/explainable_features.csv', sample_ratio=0.1, feature_studied=0, balance=True):
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

    y = y == feature_studied

    if balance:
        # Subsample the dataframe to have balanced classes
        X = pd.concat([X[y], X[~y].sample(sum(y), random_state=33)], axis=0, ignore_index=False)
        y = pd.concat([y[y], y[~y].sample(sum(y), random_state=33)], axis=0, ignore_index=False)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=33, stratify=y) #Como que 33?

    return X, y, X_train, X_test, y_train, y_test


try:
    n_gen = int(sys.argv[1])
    pop_size = int(sys.argv[2])
    nRules = int(sys.argv[3])
    nAnts = int(sys.argv[4])
    runner = int(sys.argv[5])
    feature_studied = int(sys.argv[6]) - 1
except:
    print('Using default parameters')
    n_gen = 1000
    pop_size = 30
    nRules = 15
    nAnts = 4
    runner = 1
    feature_studied = 19


fz_type_studied = fs.FUZZY_SETS.t1
checkpoints = 0

X, y, X_train, X_test, y_train, y_test = load_explainable_features(sample_ratio=1.0, feature_studied=feature_studied, balance=False)

precomputed_partitions = utils.construct_partitions(X, fz_type_studied)
#precomputed_partitions=None
min_bounds = np.min(X, axis=0).values
max_bounds = np.max(X, axis=0).values
domain = [min_bounds, max_bounds]

print('Training fuzzy classifier:' , nRules, 'rules, ', nAnts, 'ants, ', n_gen, 'generations, ', pop_size, 'population size')
fl_classifier = GA.FuzzyRulesClassifier(nRules=nRules, nAnts=nAnts, 
    linguistic_variables=precomputed_partitions, n_linguist_variables=3, 
    fuzzy_type=fz_type_studied, verbose=True, tolerance=0.0, domain=domain, runner=runner)
fl_classifier.customized_loss(new_loss)
fl_classifier.fit(X_train, y_train, n_gen=n_gen, pop_size=pop_size, checkpoints=checkpoints)

str_rules = eval_tools.eval_fuzzy_model(fl_classifier, X_train, y_train, X_test, y_test, 
                        plot_rules=False, print_rules=True, plot_partitions=False, return_rules=True)

# Check for other rule bases in the folder
import os
res_folder = 'rules_art_features'
if not os.path.exists(res_folder):
    os.makedirs(res_folder)

files = os.listdir('rules_art_features')
# We save the rules in a file
with open('rules_art_features/rules_feature' + str(feature_studied) + '.txt', 'w') as f:
    f.write(str_rules)

print('Done')
