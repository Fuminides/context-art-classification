import pandas as pd
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

pca = PCA(n_components=2, whiten=True)

deep_features = pd.read_csv('./Infus_experiments_rules_data/train_deep_features.csv', index_col=0)
deep_features = softmax(deep_features.values)

import matplotlib.pyplot as plt
import numpy as np

means = deep_features.mean(axis=0)
stds = deep_features.std(axis=0)
plt.bar(range(20), means); plt.xticks(range(20));
plt.savefig('./Infus_experiments_rules_data/means.pdf')
X_pca = pca.fit_transform(deep_features)
X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(deep_features)

bigger_than_average = deep_features > deep_features.mean(axis=1, keepdims=True)*1.4
print('Average: number of dominant features: ', bigger_than_average.sum(axis=1).mean())
plt.figure()
plt.scatter(X_embedded[:,0], X_embedded[:,1])
plt.show()
print('Escape')