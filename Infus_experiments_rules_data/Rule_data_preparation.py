'''
Use this script to prepare the data for the rules experiments.

author: @fuminides (Javier Fumanal Idocin)
'''
import pandas as pd
from scipy.special import softmax

style_train = pd.read_csv('Infus_experiments_rules_data/style_predictions_val.csv', index_col=0)
style_val = pd.read_csv('Infus_experiments_rules_data/style_predictions_train.csv', index_col=0)
style_test = pd.read_csv('Infus_experiments_rules_data/style_predictions_test.csv', index_col=0)

grad_train = pd.read_csv('Infus_experiments_rules_data/GradCam_feats.csv', index_col='Image')
grad_train.drop('Unnamed: 0', axis=1, inplace=True)
grad_train.index = grad_train.index.str.replace('.csv', '')
# Erase the elements in grad_train that are not in style_train
grad_train = grad_train.loc[style_train.index]

aux = pd.concat([style_train, grad_train], axis=1, ignore_index=False)

deep_features_full = pd.read_csv('Infus_experiments_rules_data/train_deep_features.csv', index_col=0)

# deep_features_full = pd.concat([deep_features_train, deep_features_val], axis=0, ignore_index=False)
deep_features_full = deep_features_full.loc[aux.index]


bigger_than_average = deep_features_full.values.argmax(axis=1)
aux['Dominant_feature'] = bigger_than_average

aux.to_csv('Infus_experiments_rules_data/explainable_features.csv')

print('Files formated')