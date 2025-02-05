from sklearn.datasets import (load_iris,
                              load_wine,
                              load_breast_cancer,
                              load_digits)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score as sil,
                             davies_bouldin_score as db,
                             calinski_harabasz_score as ch)
import matplotlib.pyplot as plt
from adjusted_entropy import *
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score as nmi

def get_labels_list(X, y, max_k=None):
    n = len(X)
    if max_k == None:
        k_list = range(2, n)
    else:
        k_list = range(2, min(n, max_k))
    labels_list = []
    for k in k_list:
        labels = KMeans(n_clusters=k, random_state=0).fit(X).labels_
        labels_list.append(labels)
    return labels_list

def neg_db(X, labels):
    return -db(X, labels)

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
            plt.savefig(key + '.png')
            plt.close()
        models = sorted(value, key = lambda x: x['score'], reverse = True)
        best_model = models[0]
        #best_model = get_first_peak(value)
        best_models.append({'name': key,
                            'k': best_model['k'],
                            'labels': best_model['labels']})
    return best_models

def select_models(X, y, max_k=None):
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

    scores = {#'ae_num': adjusted_entropy_num,
              #'ae_all': adjusted_entropy_all,
              'entr': entropy,
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

datasets = {'iris': load_iris(return_X_y = True),
            'digits': load_digits(return_X_y = True),
            'wine': load_wine(return_X_y = True),
            'breast cancer': load_breast_cancer(return_X_y = True)}

scores = []
for key, value in datasets.items():
    #if key != 'digits':
    #    continue
    print(key)
    X, y = value
    k = len(set(y))
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    models = select_models(X, y, 15)
    scores.append(nmi(k, models))
    scores[-1]['dataset'] = key

print(scores)
#df = pd.DataFrame(scores)
#df = df.drop(columns=['k'])
#df = df.set_index(['dataset'])
#df = df.T
#plt.figure()
#df.plot.bar(stacked=True)
#plt.ylabel('error')
#plt.savefig('datasets_stacked.png', dpi=300)
#df.plot.bar(subplots=True, legend=False, sharey=True,
#            title=['' for x in range(8)])
#plt.savefig('datasets_subplots.png', dpi=300)
#df['mean'] = df.mean(axis=1)
#df.to_csv('datasets.csv')
