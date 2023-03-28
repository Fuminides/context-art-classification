import pandas as pd
from scipy.special import softmax

style_train = softmax(pd.read_csv('Infus_experiments_rules_data/style_predictions_val.csv', index_col=0), axis=1)
style_val = softmax(pd.read_csv('Infus_experiments_rules_data/style_predictions_train.csv', index_col=0), axis=1)
style_test = softmax(pd.read_csv('Infus_experiments_rules_data/style_predictions_test.csv', index_col=0), axis=1)

grad_train = pd.read_csv('Infus_experiments_rules_data/train_grad_features.csv', index_col=0)
grad_train.columns = ['Small', 'Large', 'One or two', 'Few', 'Some', 'Lots']

grad_train[['Small', 'Large']]

aux = pd.concat([style_train, grad_train], axis=1)

grad_train[['Small', 'Large']] = softmax(grad_train[['Small', 'Large']], axis=1)
grad_train[['One or two', 'Few', 'Some', 'Lots']] = softmax(grad_train[['One or two', 'Few', 'Some', 'Lots']], axis=1)

print('Files formated')