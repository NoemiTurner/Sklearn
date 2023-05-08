# TODO: your reusable general-purpose functions here

import operator
import numpy as np

def randomize_in_place(alist, parallel_list=None):
    for i in range(len(alist)):
        # generate a random index to swap this value at i with
        # rand int in [0, len(alist))
        rand_index = np.random.randint(0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] =\
                parallel_list[rand_index], parallel_list[i]


def compute_euclidean_distance(v1, v2):
    return np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
        
        

def normalize_value(X_train, index, value):
    column = []
    for row in X_train:
        v = row[index]
        column.append(v)

    min_val = min(column)
    max_val = max(column)
    range_val = max_val - min_val

    normalized_val = (value - min_val) / range_val

    return normalized_val
