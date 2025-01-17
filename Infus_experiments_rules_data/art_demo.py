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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Matthew Correlation Coefficient
from sklearn.metrics import matthews_corrcoef
# Supress warnings
import warnings
warnings.filterwarnings("ignore")

def plot3d(X_pca, y):
    # Plot the data in 3d
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], X_pca[y == 0, 2], c='r', label='Suppresed')
    ax.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], X_pca[y == 1, 2], c='b', label='Dominant')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.legend()
    # plt.show()

def plot2d(X_pca, y):
    # Plot the data in 3d
    plt.figure(figsize=(10, 10))
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], c='r', label='Suppresed')
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c='b', label='Dominant')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    # plt.show()

def plot_PCA(n_components, X, y):
    # Visualize the data using PCA
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    if n_components == 3:
        plot3d(X_pca, y)
    else:
        plot2d(X_pca, y)

def balanced_sample(X, y):
    '''
    Create a balanced random partition of the data to evaluate the rule base.

    :param X: array of train samples. X shape = (n_samples, n_features)
    :param y: array of train labels. y shape = (n_samples,)
    :return: X_balanced, y_balanced
    '''
    # Susbsample X and y to have balanced classes
    positive_index = np.where(y)[0]
    negative_index = np.where(y == 0)[0]

    negative_samples_index = np.random.choice(negative_index, len(positive_index), replace=False)
    total_index = np.concatenate((positive_index, negative_samples_index))
    np.random.shuffle(total_index)

    X_balanced = X[total_index]
    y_balanced = y[total_index]

    return X_balanced, y_balanced

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
        X_balanced, y_balanced = balanced_sample(X, y)
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
    n_gen = 100
    pop_size = 30
    nRules = 15
    nAnts = 4
    runner = 1
    feature_studied = 16


fz_type_studied = fs.FUZZY_SETS.t1
checkpoints = 0
'''
for feature_studied in range(20):
    X, y, X_train, X_test, y_train, y_test = load_explainable_features(sample_ratio=1.0, feature_studied=feature_studied, balance=False)
    plot_PCA(2, X, y)
    plt.savefig('PCA_2D_{}.jpg'.format(feature_studied))
'''
X, y, X_train, X_test, y_train, y_test = load_explainable_features(sample_ratio=1.0, feature_studied=feature_studied, balance=False)
# plot_PCA(2, X, y)
# plt.show()
precomputed_partitions = utils.construct_partitions(X, fz_type_studied)
#precomputed_partitions=None
min_bounds = np.min(X, axis=0).values
max_bounds = np.max(X, axis=0).values
domain = [min_bounds, max_bounds]


# Use SMOTE to upsample the minority class
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=33)
X_train, y_train = sm.fit_resample(X_train, y_train)

# Susbsample X and y to have balanced classes
X_train_balanced, y_train_balanced = balanced_sample(np.array(X_test), np.array(y_test))
X_balanced, y_balanced = balanced_sample(np.array(X_test), np.array(y_test))

# Gradient boosting classification for comparison
print('Training gradient boosting classifier')
gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=33)
gb_classifier.fit(X_train_balanced, y_train_balanced)
print('Accuracy of the gradient boosting classifier on training set: {:.2f}'.format(gb_classifier.score(X_train_balanced, y_train_balanced)))
print('Matthews correlation coefficient of the gradient boosting classifier on training set: {:.2f}'.format(matthews_corrcoef(y_train_balanced, gb_classifier.predict(X_train_balanced))))
print('Accuracy of the gradient boosting classifier on test set: {:.2f}'.format(gb_classifier.score(X_test, y_test)))
print('Matthews correlation coefficient of the gradient boosting classifier on test set: {:.2f}'.format(matthews_corrcoef(y_test, gb_classifier.predict(X_test))))

print('Training fuzzy classifier:' , nRules, 'rules, ', nAnts, 'ants, ', n_gen, 'generations, ', pop_size, 'population size, ', X_train.shape[0], 'samples')
fl_classifier = GA.FuzzyRulesClassifier(nRules=nRules, nAnts=nAnts, 
    linguistic_variables=precomputed_partitions, n_linguist_variables=3, 
    fuzzy_type=fz_type_studied, verbose=True, tolerance=0.0, domain=domain, runner=runner)
# fl_classifier.customized_loss(new_loss)
fl_classifier.fit(X_train, y_train, n_gen=n_gen, pop_size=pop_size, checkpoints=checkpoints)


str_rules = eval_tools.eval_fuzzy_model(fl_classifier, X_train, y_train, X_balanced, y_balanced, 
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
