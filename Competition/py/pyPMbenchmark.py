# -*- coding: utf-8 -*-
"""

 STAT640 Competition
 python script for Profile Mean benchmark

Created on Sun Sep 27 21:32:05 2015

@author: yusong
"""


# STAT640 Competition
# python script for Profile Mean benchmark

import numpy as np 
import scipy as sp 
from scipy import sparse

rpath = r'/projects/stat640/Fall2015_Data/ratings.csv'
idmap_path = r'/projects/stat640/Fall2015_Data/IDMap.csv'
ratings = np.genfromtxt(rpath, delimiter=',', skip_header=1)
idmap = np.genfromtxt(idmap_path, delimiter=',', skip_header=1)
idmap.astype(int)
# csr_matrix
# csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
# where data, row_ind and col_ind satisfy the relationship 
# a[row_ind[k], col_ind[k]] = data[k].

n_users = 10000
n_profiles = 10000
# shape = (n_users, n_profiles)
rmat_sparse = sparse.csr_matrix((ratings[:,2], (ratings[:,0]-1, ratings[:,1]-1)),
	shape=(n_users, n_profiles))
rmat = rmat_sparse.toarray()


Pnum = np.sum((rmat!=0), axis=0)
Psum = np.sum(rmat, axis=0)
Pmeans = Psum / Pnum

pred = Pmeans[idmap.astype(int)[:,1]-1]
result = np.c_[idmap.astype(int)[:,2], pred]

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# Todo : write to fomatted csv file
headers = 'ID,Prediction'
fname = r'/home/ys43/pyPMbenchmark.csv'
print "ready to write!"
np.savetxt(fname, result, delimiter=',', header= headers, fmt='%.14f', comments='')
#
## Make a prediction
#gt_path = r'/Users/yusong/Code/STAT640/Competition/data/groundTruth.csv'
#groundTruth = np.genfromtxt(gt_path, delimiter=',',skip_header=1)
#pmRMSE = rmse(pred, groundTruth[:,1])
