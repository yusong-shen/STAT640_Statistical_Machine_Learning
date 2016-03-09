"""
kfold function
@author : yusong
"""

import random 
def kfold(n, n_folds):
    """
    return the indices
    """
    result = []
    indices = range(n)
    random.shuffle(indices)
    # the first n%n_fold have size n // n_folds + 1
    fold_sizes = [(n // n_folds) for i in range(n_folds)]
    fold_sizes[:(n % n_folds)] = [(n // n_folds) + 1 for i in range(n % n_folds)]
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_ind = indices[start:stop]
        train_ind = [x for x in indices if x not in test_ind]
        result.append((train_ind, test_ind))
        current = stop
    return result

kf = kfold(15, 4)
for train_ind, test_ind in kf:
    print train_ind, test_ind
 # print    
#[8, 14, 13, 5, 10, 3, 9, 11, 0, 4, 2] [12, 6, 1, 7]
#[12, 6, 1, 7, 10, 3, 9, 11, 0, 4, 2] [8, 14, 13, 5]
#[12, 6, 1, 7, 8, 14, 13, 5, 0, 4, 2] [10, 3, 9, 11]
#[12, 6, 1, 7, 8, 14, 13, 5, 10, 3, 9, 11] [0, 4, 2]
