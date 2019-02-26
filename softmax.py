import numpy as np


def softmax(l):
    expl = np.exp(l)
    sumexpl = sum(expl)
    result = []
    for x in l:
        result.append(x*1.0/sumexpl)
    return result


def  cross_entropy(y,p):
    y = np.float_(y)
    p = np.float_(p)
    return -np.sum(y*np.log(p) + (1 - y)*np.log(1 - p))



print(softmax([2, 5, 0]))