'''
File to perform bow enconding of semart annotations.

'''

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def bow_load_train_text_corpus(semart_path='../SemArt/', k=10, append=False):
    semart_train = pd.read_csv(semart_path + 'semart_train.csv', encoding = "ISO-8859-1", sep='\t')
    semart_val = pd.read_csv(semart_path + 'semart_val.csv', encoding = "ISO-8859-1", sep='\t')
    semart_test = pd.read_csv(semart_path + 'semart_test.csv', encoding="ISO-8859-1", sep='\t')

    transformer = CountVectorizer(stop_words='english')
    transformer = transformer.fit(semart_train['DESCRIPTION'])

    coded_semart_train = transformer.transform(semart_train['DESCRIPTION'])
    coded_semart_val = transformer.transform(semart_val['DESCRIPTION'])
    coded_semart_test = transformer.transform(semart_test['DESCRIPTION'])

    freqs = np.asarray(coded_semart_train.sum(axis=0))
    bool_freqs = freqs > k

    chosen_coded_semart_train = coded_semart_train[:, bool_freqs.squeeze()]
    chosen_coded_semart_val = coded_semart_val[:, bool_freqs.squeeze()]
    chosen_coded_semart_test = coded_semart_test[:, bool_freqs.squeeze()]

    if not append:
        return chosen_coded_semart_train
    else:
        return chosen_coded_semart_train, chosen_coded_semart_val, chosen_coded_semart_test


def tf_idf_load_train_text_corpus(semart_path='../SemArt/', k=10, append=False):
    from sklearn.feature_extraction.text import TfidfVectorizer

    semart_train = pd.read_csv(semart_path + 'semart_train.csv', encoding = "ISO-8859-1", sep='\t')
    semart_val = pd.read_csv(semart_path + 'semart_val.csv', encoding = "ISO-8859-1", sep='\t')
    semart_test = pd.read_csv(semart_path + 'semart_test.csv', encoding="ISO-8859-1", sep='\t')


    transformer = CountVectorizer(stop_words='english')
    transformer = transformer.fit(semart_train['DESCRIPTION'])

    coded_semart_train = transformer.transform(semart_train['DESCRIPTION'])
    coded_semart_val = transformer.transform(semart_val['DESCRIPTION'])
    coded_semart_test = transformer.transform(semart_test['DESCRIPTION'])

    freqs = np.asarray(coded_semart_train.sum(axis=0))
    bool_freqs = freqs > k

    corpus = list(semart_train['DESCRIPTION'])
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    chosen_coded_semart_train = coded_semart_train[:, bool_freqs.squeeze()]
    chosen_coded_semart_val = coded_semart_val[:, bool_freqs.squeeze()]
    chosen_coded_semart_test = coded_semart_test[:, bool_freqs.squeeze()]

    if not append:
        return chosen_coded_semart_train
    else:
        return chosen_coded_semart_train, chosen_coded_semart_val, chosen_coded_semart_test

def fcm_coded_context(chosen_coded_semart, clusters):
    from skfuzzy.cluster import cmeans

    cntr, u, u0, d, jm, p, fpc = cmeans(chosen_coded_semart.toarray().T, clusters, 2, 0.01, 200)

    return u.T

def myplot(score,coeff,labels=None):
    import matplotlib.pyplot as plt
    plt.figure()
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]

    plt.scatter(xs ,ys, c = labels) #without scaling
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')

def silhoutte_progress():
    from sklearn.metrics import silhouette_score

    chosen_coded_semart_train = bow_load_train_text_corpus(k=10)
    silhouttes = []
    for cluster in np.arange(2, 200):
        sol = fcm_coded_context(chosen_coded_semart_train, cluster)
        crips_sol = np.argmax(sol, axis=1)
        silhouttes.append(silhouette_score(chosen_coded_semart_train, crips_sol))
    
    return silhouttes


def biplot():
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    chosen_coded_semart_train = fcm_coded_context(
                    bow_load_train_text_corpus(k=10, append=False), clusters=128)
    pca = PCA()
    pca.fit(chosen_coded_semart_train)
    x_new = pca.transform(chosen_coded_semart_train)
    myplot(x_new[:, 0:2], pca.components_)

if __name__ == '__main__':
    biplot()