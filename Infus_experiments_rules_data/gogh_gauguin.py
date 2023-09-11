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
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

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

    if len(positive_index) > len(negative_index):
        aux = positive_index
        positive_index = negative_index
        negative_index = aux

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
        # Susbsample X and y to have balanced classes
        X_balanced, y_balanced = balanced_sample(X, y)

        ev_object = evr.evalRuleBase(ruleBase, X, y)
        ev_object.add_rule_weights()

        # Get the biggest rule weight
        max_weight = max(ruleBase.get_scores())
        score_acc = ev_object.classification_eval()
        if score_acc < 0:
            score_acc = 0
        score_size = ev_object.size_eval(tolerance)

        alpha = 0.99
        beta = 0.00
        theta = 1 - alpha - beta

        score = score_acc * alpha + (1 - max_weight) * beta + score_size * theta

        return 1 - score


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
try:
    semart_path = 'C:/Users/javier.fumanal/Documents/GitHub/SemArt/'
    semart_train = pd.read_csv(semart_path + 'semart_train.csv', encoding='cp1252', sep='\t')

except FileNotFoundError:
    try:
        semart_path = '/home/javierfumanal/Documents/GitHub/SemArt/'
        semart_train = pd.read_csv(semart_path + 'semart_train.csv', encoding='cp1252', sep='\t')
    except FileNotFoundError:
        semart_path = '/home/fcojavier.fernandez/Github/SemArt/'
        semart_train = pd.read_csv(semart_path + 'semart_train.csv', encoding='cp1252', sep='\t')


van_gogh = semart_train[semart_train['AUTHOR'] == 'GOGH, Vincent van']
paul = semart_train[semart_train['AUTHOR'] == 'GAUGUIN, Paul']




try:
    n_gen = int(sys.argv[1])
    pop_size = int(sys.argv[2])
    nRules = int(sys.argv[3])
    nAnts = int(sys.argv[4])
    runner = int(sys.argv[5])
except:
    print('Using default parameters')
    n_gen = 500
    pop_size = 30
    nRules = 8
    nAnts = 3
    runner = 2


fz_type_studied = fs.FUZZY_SETS.t1
checkpoints = 0

X, y, _, _, _, _ = load_explainable_features(sample_ratio=1.0)
X_van_gogh = X.loc[van_gogh['IMAGE_FILE']]
X_paul = X.loc[paul['IMAGE_FILE']]
y_artists = np.zeros(X_van_gogh.shape[0] + X_paul.shape[0])
y_artists[:X_van_gogh.shape[0]] = 1

# Assign names to classes
y_artists = pd.Series(y_artists)
y_artists = y_artists.replace({1: 'Van Gogh', 0: 'Paul Gauguin'})
painter = 'GOGH_GAUGUIN'

X_artists = pd.concat([X_van_gogh, X_paul])

'''
# TSN visualization of X_artists
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
X_artists_tsne = tsne.fit_transform(X_artists)
X_artists_tsne = pd.DataFrame(X_artists_tsne, columns=['TSNE1', 'TSNE2'])
X_artists_tsne['AUTHOR'] = y_artists
# Plot the TSNE
import seaborn as sns
sns.scatterplot(x='TSNE1', y='TSNE2', hue='AUTHOR', data=X_artists_tsne)
plt.show()

plt.figure()
# PCA visualization of X_artists
pca = PCA(n_components=2)
X_artists_pca = pca.fit_transform(X_artists)
X_artists_pca = pd.DataFrame(X_artists_pca, columns=['PC1', 'PC2'])
X_artists_pca['AUTHOR'] = y_artists
# Plot the PCA
import seaborn as sns
sns.scatterplot(x='PC1', y='PC2', hue='AUTHOR', data=X_artists_pca)
plt.show()
'''
# Solve the problem using a Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



X_artists_train, X_artists_test, y_artists_train, y_artists_test = train_test_split(X_artists, y_artists, test_size=0.20, random_state=33, stratify=y_artists) #Como que 33?
# Print the shape of the training and test sets
print('Shape of the training set: ' + str(X_artists_train.shape))
print('Shape of the test set: ' + str(X_artists_test.shape))


from sklearn.metrics import matthews_corrcoef

classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
classifier.fit(X_artists_train, y_artists_train)
y_pred = classifier.predict(X_artists_test)
print('Accuracy of the classifier: ', accuracy_score(y_artists_test, y_pred))
# Feature importance
importances = classifier.feature_importances_
importance_threshold = 0.025
print('Number of features used: ' + str(np.sum(importances > importance_threshold)))
# Get only the important features

X = X.iloc[:, importances > importance_threshold]
X_artists_train = X_artists_train.iloc[:, importances > importance_threshold]
X_artists_test = X_artists_test.iloc[:, importances > importance_threshold]

classifier.fit(X_artists_train, y_artists_train)
y_pred = classifier.predict(X_artists_test)
print('Accuracy of the classifier with reduced dimensions: ', accuracy_score(y_artists_test, y_pred))
print('Matthews correlation coefficient: ', matthews_corrcoef(y_artists_test, y_pred))
precomputed_partitions = utils.construct_partitions(X, fz_type_studied)
#precomputed_partitions=None
min_bounds = np.min(X, axis=0).values
max_bounds = np.max(X, axis=0).values
domain = [min_bounds, max_bounds]


print('Training fuzzy classifier:' , nRules, 'rules, ', nAnts, 'ants, ', n_gen, 'generations, ', pop_size, 'population size')
fl_classifier = GA.FuzzyRulesClassifier(nRules=nRules, nAnts=nAnts, 
    linguistic_variables=precomputed_partitions, n_linguist_variables=3, 
    fuzzy_type=fz_type_studied, verbose=True, tolerance=0.001, domain=domain, runner=runner)
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




