from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score as sil,
                             davies_bouldin_score as db,
                             calinski_harabasz_score as ch)
import itertools
import numpy as np
import matplotlib.pyplot as plt
from adjusted_entropy import *

def cluster(X, k):
    model = KMeans(n_clusters=k, random_state=0).fit(X)
    #model = AgglomerativeClustering(n_clusters=k, linkage='single').fit(X)
    return model.labels_

def get_labels_list(X, y, max_k=None):
    n = len(X)
    if max_k == None:
        k_list = range(2, n)
    else:
        k_list = range(2, min(n, max_k))
    labels_list = []
    for k in k_list:
        labels_list.append(cluster(X, k))
    return labels_list

def neg_db(X, labels):
    return -db(X, labels)

def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def get_first_peak(d):
    for i, j in pairwise(d):
        if i['score'] > j['score']:
            return i
    return j

def optimize(labels_list, scores, X=[], plot=False):
    model_lists = {}
    for key, value in scores.items():
        model_lists[key] = []
    for labels in labels_list:
        for key, value in model_lists.items():
            score_foo = scores[key]
            if len(X) != 0:
                score_value = score_foo(X, labels)
            else:
                score_value = score_foo(labels)
            value.append({'k': len(set(labels)),
                          'labels': labels,
                          'score': score_value})
    best_models = []
    for key, value in model_lists.items():
        if plot:
            k_list = [x['k'] for x in value]
            score_list = [x['score'] for x in value]
            plt.figure()
            plt.plot(k_list, score_list)
            plt.savefig('real/' + key + '.png')
            plt.close()
        #models = sorted(value, key = lambda x: x['score'], reverse = True)
        #best_model = models[0]
        best_model = get_first_peak(value)
        best_models.append({'name': key,
                            'k': best_model['k'],
                            'labels': best_model['labels']})
    return best_models

def select_models(X, y, max_k=None):
    print(part(y))
    scores = {'sil': sil, 'db': neg_db, 'ch': ch}
    labels_list = get_labels_list(X, y, max_k)
    models = optimize(labels_list, scores, X, True)

    selected = []

    model = next(item for item in models if item['name'] == 'sil')
    selected.append({'name': 'sil', 'k': model['k']})

    model = next(item for item in models if item['name'] == 'db')
    selected.append({'name': 'db', 'k': model['k']})

    model = next(item for item in models if item['name'] == 'ch')
    selected.append({'name': 'ch', 'k': model['k']})

    scores = {'ae_num': adjusted_entropy_num,
              'ae_all': adjusted_entropy_all,
              'entr': entr,
              'ne_num': normalized_entropy_num,
              'ne_all': normalized_entropy_all,}
    labels_list = [x['labels'] for x in models]
    selected += optimize(labels_list, scores)

    return selected

def abs_err(k_true, k_pred):
    return abs(k_true - k_pred)

def evaluate(true_k, models):
    scores = {'k': true_k}
    for model in models:
        scores[model['name']] = abs_err(model['k'], true_k)
    return scores
