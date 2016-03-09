# -*- coding: utf-8 -*-
"""
Collaborate Filtering for STAT640 Competition
Created on Thu Nov 26 22:15:08 2015

@author: yusong
"""


import numpy as np 
import scipy as sp 
from scipy import sparse
import time
from scipy.optimize import minimize
from cofiHelper import *

# Measure the running time
start = time.time()

Nfeval = 1

def callbackCofi(Xi):
    global Nfeval
    print '{0:4d}'.format(Nfeval)
    Nfeval += 1

###############################################################################
rpath = r'/Users/yusong/Code/STAT640/Competition/data/ratings.csv'
idmap_path = r'/Users/yusong/Code/STAT640/Competition/data/IDMap.csv'
kaggleid_path = r'/Users/yusong/Code/STAT640/Competition/data/KaggleID.csv'

ratings = np.genfromtxt(rpath, delimiter=',', skip_header=1)
idmap = np.genfromtxt(idmap_path, delimiter=',', skip_header=1)
idmap.astype(int)


debug = False
n_users = 10000
n_profiles = 10000
# Note : shape = (n_profiles, n_users)
rmat_sparse = sparse.csr_matrix((ratings[:,2], (ratings[:,1]-1, ratings[:,0]-1)),
	shape=(n_profiles, n_users))
Y = rmat_sparse.toarray()
R = (Y!=0)
if debug:
    Y = Y[:,:500]
    R = (Y!=0)
    n_users = Y.shape[1]
    n_profiles = Y.shape[0]
    
# Split the dataset
test_size = 0.2
Y_train, Y_test, R_train, R_test = splitDataset(Y, R, test_size)


#  normalize rating
Ynorm_train, Ymean_train = normalizeRatings(Y_train, R_train);


n_features = 10;

# Set Initial Parameters (Theta, X)
X = np.random.randn(n_profiles, n_features) 
Theta = np.random.randn(n_users, n_features) 

init_params = np.r_[X.flatten(), Theta.flatten()] #?
# parameter values passed tk cofiCost and cofiGrad
lambdaa = 10
# should be Ynorm or Y?
args = (Ynorm_train, R_train, n_users, n_profiles, n_features, lambdaa)

# Set options for minimize function
opts = {'maxiter' : 1,
        'disp' : True
        }
if debug:
    opts['maxiter'] = 1
        
theta = minimize(cofiCost,
                 init_params, jac=cofiGrad, args=args,
                 method='CG', options=opts,
#                 callback = callbackCofi,
                 )

# Unfold the returned theta back into U and W
X = np.reshape(theta.x[:n_profiles*n_features], (n_profiles, n_features))
Theta = np.reshape(thet.x[n_profiles*n_features:], (n_users, n_features))


###############################################################################
p = np.dot(X, Theta.T)
# repeat the Ymean column n_users time 
pred = p + np.tile(Ymean,(1,n_users))
tr_pred = np.multiply(pred, R_train)
tr_rmse = rmse(pred, Y_train)
print "training set rmse : %s"%(tr_rmse)


# choose those entries we want
ts_pred = np.multiply(pred, R_test)
ts_rmse = rmse(pred, Y_test)
print "test set rmse : %s"%(ts_rmse)

###############################################################################
#
## Todo : write to fomatted csv file
#headers = 'ID,Prediction'
#fname = r'/home/ys43/pyPMbenchmark.csv'
#print "ready to write!"
#np.savetxt(fname, result, delimiter=',', header= headers, fmt='%.14f', comments='')
#

end = time.time()
print "time elapse : %6.2f secondes"%(end-start)