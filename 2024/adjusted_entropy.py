import numpy as np
from scipy.stats import entropy
from scipy.special import binom, stirling2, comb
from math import log

#Utils
def part(y):
    value, counts = np.unique(y, return_counts = True)
    return counts   

def bell(n):
    b = 0
    for i in range(0, n + 1):
        b += stirling2(n, i)
    return b

def w_inf(c, n, base = 2):
    return -(c/n)*log(c/n, base)
    #return -log(c/n, base)

def n_num(n, k, c):
    return binom(n, c)*stirling2(n - c, k - 1)

def p_num(n, k, c):
    return n_num(n, k, c)/stirling2(n, k)

def p_all(n, c, b):
    #return binom(n, c)*bell(n - c)/b
    return binom(n, c)*bell(n - c)/b

def adj(v, exp_v, max_v):
    if max_v - exp_v == 0:
        return None
    return (v - exp_v)/(max_v - exp_v)

def normalized_entropy_num(y):
    p = part(y)
    k = len(p)
    #print(log(k))
    return entropy(p)/log(k)

def normalized_entropy_all(y):
    n = len(y)
    p = part(y)
    return entropy(p)/log(n)

#Expected values
def exp_entr_num(n, k, base = 2):
    exp_e = 0
    for c in range(1, n - k + 2):
        e = w_inf(c, n, base)
        p = p_num(n, k, c)
        exp_e += p*e
    return exp_e

def exp_entr_all(n, base = 2):
    exp_e = 0
    b = bell(n)
    #for k in range(1, n + 1):
    #    for c in range(1, n - k + 2):
    #        e = w_inf(c, n, base)
    #        w = w_all(n, k, c, b)
    #        exp_e += w*e
    for c in range(1, n + 1):
        e = w_inf(c, n, base)
        print('e', e)
        p = p_all(n, c, b)
        print('p', p)
        exp_e += p*e
    return exp_e

#Adjustment
def adjusted_entropy_num(y, base = 2):
    n = len(y)
    p = part(y)
    k = len(p)
    e = entropy(p, base = base)
    max_e = log(k, base)
    exp_e = exp_entr_num(n, k, base)
    return adj(e, exp_e, max_e)

def adjusted_entropy_all(y, base = 2):
    n = len(y)
    p = part(y)
    k = len(p)
    e = entropy(p, base = base)
    max_e = log(n, base)
    exp_e = exp_entr_all(n, base)
    return adj(e, exp_e, max_e)

#y = [0, 0, 1, 2]
#print(normalized_entropy_num(y))
#print(adjusted_entropy_num(y))
