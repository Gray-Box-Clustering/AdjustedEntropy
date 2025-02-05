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


datasets = {#'iris': load_iris(return_X_y = True)
            'digits': load_digits(return_X_y = True)
            #'wine': load_wine(return_X_y = True)
            #'breast cancer': load_breast_cancer(return_X_y = True)
            }

scaler = StandardScaler()
scores = []
for key, value in datasets.items():
    X, y = value
    n = len(X)
    k = len(set(y))
    X = scaler.fit_transform(X)
    k_list = range(2, int(n/k)+1)
    for k in k_list:
        labels = KMeans(n_clusters=k, random_state=0).fit(X).labels_
        p = part(labels)
        scores.append({
            'k': len(set(labels)),
            'sil': sil(X, labels),
            'db': -db(X, labels),
            'ch': ch(X, labels),
            'e': entropy(p),
            'ne_num': normalized_entropy_num(labels),
            'ne_all': normalized_entropy_all(labels),
            'nmi': nmi(y, labels)})

plt.figure()
k_list = [x['k'] for x in scores]
sil_list = [x['sil'] for x in scores]
db_list = [x['db'] for x in scores]
ch_list = [x['ch'] for x in scores]
e_list = [x['e'] for x in scores]
ne_num = [x['ne_num'] for x in scores]
ne_all = [x['ne_all'] for x in scores]
nmi = [x['nmi'] for x in scores]
plt.plot(k_list, sil_list, label = 'sil')
plt.plot(k_list, db_list, label = 'db')
#plt.plot(k_list, ch_list, label = 'ch')
plt.plot(k_list, e_list, label = 'e')
plt.plot(k_list, ne_num, label = 'ne_num')
plt.plot(k_list, ne_all, label = 'ne_all')
plt.plot(k_list, nmi, label = 'nmi')
plt.legend()
plt.show()

k_true = len(set(y))
best_sil = sorted(scores, key = lambda x: x['sil'], reverse = True)[0]
best_db = sorted(scores, key = lambda x: x['db'], reverse = True)[0]
best_ch = sorted(scores, key = lambda x: x['ch'], reverse = True)[0]
k_sil = sorted(scores, key = lambda x: x['sil'], reverse = True)[0]['k']
k_db = sorted(scores, key = lambda x: x['db'], reverse = True)[0]['k']
k_ch = sorted(scores, key = lambda x: x['ch'], reverse = True)[0]['k']
bests = [best_sil, best_db, best_ch]
best_e = sorted(bests, key = lambda x: x['e'], reverse = True)[0]
best_num = sorted(bests, key = lambda x: x['ne_num'], reverse = True)[0]
best_all = sorted(bests, key = lambda x: x['ne_all'], reverse = True)[0]
k_e = best_e['k']
k_num = best_num['k']
k_all = best_all['k']
k_nmi = sorted(scores, key = lambda x: x['nmi'], reverse = True)[0]['k']
print('sil', abs(k_sil-k_true))
print('db', abs(k_db-k_true))
print('ch', abs(k_ch-k_true))
print('e', abs(k_e-k_true))
print('ne_num', abs(k_num-k_true))
print('ne_all', abs(k_all-k_true))
print('nmi', abs(k_nmi-k_true))
print('sil_nmi', best_sil['nmi'])
print('db_nmi', best_db['nmi'])
print('ch_nmi', best_ch['nmi'])
print('e_nmi', best_e['nmi'])
print('num_nmi', best_num['nmi'])
print('all_nmi', best_all['nmi'])
    
    
