import pandas as pd
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt

deep_features = pd.read_csv('./Infus_experiments_rules_data/train_deep_features.csv', index_col=0)
deep_features = softmax(deep_features.values)

import matplotlib.pyplot as plt
import numpy as np

means = deep_features.mean(axis=0)
stds = deep_features.std(axis=0)
plt.bar(range(20), means, yerr=stds); plt.xticks(range(20));
plt.savefig('./Infus_experiments_rules_data/means.pdf')

bigger_than_average = deep_features > deep_features.mean(axis=1, keepdims=True)*1.4
print('Average: number of dominant features: ', bigger_than_average.sum(axis=1).mean())

print('Escape')