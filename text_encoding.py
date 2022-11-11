'''
File to perform bow enconding of semart annotations.

'''

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import clip

def bow_load_train_text_corpus(semart_path='../SemArt/', k=10, append='False', top=True, explain=False):
    semart_train = pd.read_csv(semart_path + 'semart_train.csv', encoding = "ISO-8859-1", sep='\t')
    semart_val = pd.read_csv(semart_path + 'semart_val.csv', encoding = "ISO-8859-1", sep='\t')
    semart_test = pd.read_csv(semart_path + 'semart_test.csv', encoding="ISO-8859-1", sep='\t')

    transformer = CountVectorizer(stop_words='english')
    transformer = transformer.fit(semart_train['DESCRIPTION'])

    coded_semart_train = transformer.transform(semart_train['DESCRIPTION'])
    coded_semart_val = transformer.transform(semart_val['DESCRIPTION'])
    coded_semart_test = transformer.transform(semart_test['DESCRIPTION'])

    freqs = np.asarray(coded_semart_train.sum(axis=0))

    if not top:
        bool_freqs = freqs > k

        chosen_coded_semart_train = coded_semart_train[:, bool_freqs.squeeze()]
        chosen_coded_semart_val = coded_semart_val[:, bool_freqs.squeeze()]
        chosen_coded_semart_test = coded_semart_test[:, bool_freqs.squeeze()]
        word_name = transformer.get_feature_names_out()
    else:
        sorted_freqs = np.argsort(freqs)
        chosen_words = sorted_freqs[0][::-1][0:k]

        chosen_coded_semart_train = coded_semart_train[:, chosen_words]
        chosen_coded_semart_val = coded_semart_val[:, chosen_words]
        chosen_coded_semart_test = coded_semart_test[:, chosen_words]
        word_name = transformer.get_feature_names_out()[chosen_words]

    if explain:
        if append != 'append':
            return chosen_coded_semart_train, word_name
        else:
            return chosen_coded_semart_train, chosen_coded_semart_val, chosen_coded_semart_test, word_name
    else:
        if append != 'append':
            return chosen_coded_semart_train
        else:
            return chosen_coded_semart_train, chosen_coded_semart_val, chosen_coded_semart_test

def clip_load_train_text_corpus(semart_path='../SemArt/', k=10, append='False', top=True, explain=False):
    import torch
    semart_train = pd.read_csv(semart_path + 'semart_train.csv', encoding = "ISO-8859-1", sep='\t')
    semart_val = pd.read_csv(semart_path + 'semart_val.csv', encoding = "ISO-8859-1", sep='\t')
    semart_test = pd.read_csv(semart_path + 'semart_test.csv', encoding="ISO-8859-1", sep='\t')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #model, preprocess = clip.load("ViT-B/32", device=device)

    semart_train_desc = semart_train['DESCRIPTION']
    semart_val_desc = semart_val['DESCRIPTION']
    semart_test_desc = semart_test['DESCRIPTION']

    MAX_LEN_CLIP = 77 # TODO: CACHE THESE FEATURES
    try:
        train_mat = pd.read_csv('cache/clip_train.csv', index_col=0)
        val_mat = pd.read_csv('cache/clip_val.csv', index_col=0)
        test_mat = pd.read_csv('cache/clip_test.csv', index_col=0)
    except:
        train_mat = np.zeros((len(semart_train_desc), MAX_LEN_CLIP))
        for ix, context_text in enumerate(semart_train_desc):
            slices = [context_text[i:i+MAX_LEN_CLIP] for i in range(0, len(context_text), MAX_LEN_CLIP)]
            text_train = clip.tokenize(slices).cpu().numpy().max(axis=0)
            train_mat[ix, :] = text_train

        val_mat = np.zeros((len(semart_val_desc), MAX_LEN_CLIP))
        for ix, context_text in enumerate(semart_val_desc):
            slices = [context_text[i:i+MAX_LEN_CLIP] for i in range(0, len(context_text), MAX_LEN_CLIP)]
            text_train = clip.tokenize(slices).cpu().numpy().max(axis=0)
            val_mat[ix, :] = text_train

        test_mat = np.zeros((len(semart_test_desc), MAX_LEN_CLIP))
        for ix, context_text in enumerate(semart_test_desc):
            slices = [context_text[i:i+MAX_LEN_CLIP] for i in range(0, len(context_text), MAX_LEN_CLIP)]
            text_train = clip.tokenize(slices).cpu().numpy().max(axis=0)
            test_mat[ix, :] = text_train

        pd.DataFrame(train_mat).to_csv('cache/clip_train.csv')
        pd.DataFrame(val_mat).to_csv('cache/clip_val.csv')
        pd.DataFrame(test_mat).to_csv('cache/clip_test.csv')

    return train_mat, val_mat, test_mat

def tf_idf_load_train_text_corpus(semart_path='../SemArt/', k=10, append='append', top=True, explain=False):
    from sklearn.feature_extraction.text import TfidfVectorizer

    semart_train = pd.read_csv(semart_path + 'semart_train.csv', encoding = "ISO-8859-1", sep='\t')
    semart_val = pd.read_csv(semart_path + 'semart_val.csv', encoding = "ISO-8859-1", sep='\t')
    semart_test = pd.read_csv(semart_path + 'semart_test.csv', encoding="ISO-8859-1", sep='\t')

    transformer = CountVectorizer(stop_words=None)
    transformer = transformer.fit(semart_train['DESCRIPTION'])
  

    bow_coded_semart_train = transformer.transform(semart_train['DESCRIPTION'])

    corpus = list(semart_train['DESCRIPTION'])
    val_corpus = list(semart_val['DESCRIPTION'])
    test_corpus = list(semart_test['DESCRIPTION'])

    freqs = np.asarray(bow_coded_semart_train.sum(axis=0))

    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    
  

    if not top:
        bool_freqs = freqs > k

        chosen_coded_semart_train = vectorizer.transform(corpus)[:, bool_freqs.squeeze()]
        chosen_coded_semart_val = vectorizer.transform(val_corpus)[:, bool_freqs.squeeze()]
        chosen_coded_semart_test = vectorizer.transform(test_corpus)[:, bool_freqs.squeeze()]

        word_name = transformer.get_feature_names_out()
    else:
        sorted_freqs = np.argsort(freqs)
        chosen_words = sorted_freqs[0][::-1][0:k]

        chosen_coded_semart_train = vectorizer.transform(corpus)[:, chosen_words]
        chosen_coded_semart_val = vectorizer.transform(val_corpus)[:, chosen_words]
        chosen_coded_semart_test = vectorizer.transform(test_corpus)[:, chosen_words]
        word_name = transformer.get_feature_names_out()[chosen_words]

    if explain:
        if append != 'append':
            return chosen_coded_semart_train, word_name
        else:
            return chosen_coded_semart_train, chosen_coded_semart_val, chosen_coded_semart_test, word_name
    else:
        if append != 'append':
            return chosen_coded_semart_train
        else:
            return chosen_coded_semart_train, chosen_coded_semart_val, chosen_coded_semart_test

def prune_corpus(corpus):
    joined_corpus = '$$'.join(corpus)
    joined_corpus2 = joined_corpus.split(' ')
    counts = Counter(joined_corpus2)
    correct_words = list(filter(lambda a: a[1] > 10, list(counts.items())))
    pruned_words = [b[0] for a, b in enumerate(correct_words)]

    pruned_corpus = ' '.join(filter(lambda a: a in pruned_words, joined_corpus2)).split('$$')

    return pruned_corpus

def fcm_coded_context(chosen_coded_semart, clusters):
    from skfuzzy.cluster import cmeans

    try:
        cntr, u, u0, d, jm, p, fpc = cmeans(chosen_coded_semart.T.toarray(), clusters, 2, 0.01, 200)
    except AttributeError:
        cntr, u, u0, d, jm, p, fpc = cmeans(chosen_coded_semart.T, clusters, 2, 0.01, 200)

    return u.T

def myplot(score,coeff,labels=None):
    import matplotlib.pyplot as plt
    plt.figure()
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]

    plt.figure()
    plt.scatter(xs ,ys, c = labels) #without scaling
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print('Biplot done')

    '''for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')'''

def silhoutte_progress():
    from sklearn.metrics import silhouette_score

    chosen_coded_semart_train = clip_load_train_text_corpus(k=10)
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
                    clip_load_train_text_corpus(k=10, append=False)[0], clusters=128)
    pca = PCA()
    pca.fit(chosen_coded_semart_train)
    print(pca.explained_variance_ratio_[0:2])
    x_new = pca.transform(chosen_coded_semart_train)
    myplot(x_new[:, 0:2], pca.components_)

def biplot_tfidf():
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    chosen_coded_semart_train = fcm_coded_context(
                    tf_idf_load_train_text_corpus(k=10, append=False), clusters=128)
    pca = PCA()
    pca.fit(chosen_coded_semart_train)
    x_new = pca.transform(chosen_coded_semart_train)
    myplot(x_new[:, 0:2], pca.components_)

if __name__ == '__main__':
    biplot()