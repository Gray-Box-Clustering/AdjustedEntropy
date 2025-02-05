from adjusted_entropy import *


c = [1, 2, 3, 4]
p = part(c)
n = len(c)
k = len(p)

print(entropy(p, base=2))
#print(n_num(n, 3, 3))
#print(n_num(n, 3, 2))
#print(n_num(n, 3, 1))

print(exp_entr_all(n))
for i in [1, 2, 3, 4]:
    print(exp_entr_num(n, i))
