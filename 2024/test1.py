from numpy.random import randint
from scipy.stats import entropy
from math import log
from adjusted_entropy import *
import pandas as pd
import matplotlib.pyplot as plt

n = 50
k_max = 50
k_arr = range(2, k_max + 1)

res = []
t = 1
for k in k_arr:
    item = {}
    e = 0
    e_norm_num = 0
    e_norm_all = 0
    #a_e_num = 0
    #a_e_all = 0
    for i in range(t):
        y = randint(0, k, n)
        #y = [0 for x in range(int(n/2))] + randint(0, k, int(n/2))
        p = part(y)
        #print(p)
        e = entropy(p)
        #print(e)
        e += e/t
        e_norm_num += normalized_entropy_num(y)/t
        e_norm_all += normalized_entropy_all(y)/t
        #a_e = adjusted_entropy_num(y)
        #print(a_e)
        #a_e_num += a_e/t
        #a_e_all += adjusted_entropy_all(y)/t
    res.append({'e': e,
                'e_norm_num': e_norm_num,
                'e_norm_all': e_norm_all})
                #'a_e_num': a_e_num})
                #'a_e_all': a_e_all})

df = pd.DataFrame(res, index = k_arr)
df.to_csv('test1.csv')
plt.plot(df['e'], label = 'e') 
plt.plot(df['e_norm_num'], label = 'e_norm_num') 
plt.plot(df['e_norm_all'], label = 'e_norm_all') 
#plt.plot(df['a_e_num'], label = 'a_e_num')
#plt.plot(df['a_e_all'], label = 'a_e_all') 
plt.xlabel('K')
plt.legend()
plt.show() 
