# -*- coding: utf-8 -*-
"""
STAT640 HW02 P3
14-cancer microarray data

14-cancer  gene expression data. 16,063 genes, 144 training samples,
54 test samples. 
One gene per row, one sample per column

Created on Wed Oct 14 16:17:07 2015

@author: yusong
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import lasso_path, enet_path

# read the 14-cancer microarray data

xtrain_path = r'/Users/yusong/Code/STAT640/data/14cancer.xtrain.csv'
xtest_path = r'/Users/yusong/Code/STAT640/data/14cancer.xtest.csv'
ytrain_path = r'/Users/yusong/Code/STAT640/data/14cancer.ytrain'
ytest_path = r'/Users/yusong/Code/STAT640/data/14cancer.ytest'

xtrain = np.genfromtxt(xtrain_path, delimiter=',', skip_header=1)
xtrain = np.transpose(xtrain)
xtest = np.genfromtxt(xtest_path, delimiter=',', skip_header=1)
xtest = np.transpose(xtest)
ytrain = np.genfromtxt(ytrain_path, delimiter=' ', skip_header=0)
ytest = np.genfromtxt(ytest_path, delimiter=' ', skip_header=1)

## ===========
## ridge regression
ridge_clf = linear_model.RidgeCV(alphas=np.logspace(-4,2,200))
ridge_clf.fit(xtrain, ytrain)

# predict
ridge_pred = ridge_clf.predict(xtest)
ridge_ts_err = 1 - ridge_clf.score(xtest, ytest)
ridge_tr_err = 1 - ridge_clf.score(xtrain, ytrain)

# show result
print("ridge regression classification training error: %s \n testing error: %s \n"
    %(ridge_tr_err,ridge_ts_err))
    

## lasso regression
lasso_clf = linear_model.LassoCV(alphas=np.logspace(-4,-2,200))
lasso_clf.fit(xtrain, ytrain)

# predict
lasso_pred = lasso_clf.predict(xtest)
lasso_ts_err = 1 - lasso_clf.score(xtest, ytest)
lasso_tr_err = 1 - lasso_clf.score(xtrain, ytrain)

# show result
print("lasso regression classification training error: %s \n testing error: %s \n"
    %(lasso_tr_err,lasso_ts_err))    
    
    
## elasticNet regression
elasticNet_clf = linear_model.ElasticNetCV(alphas=np.logspace(-4,-2,200))
elasticNet_clf.fit(xtrain, ytrain)

# predict
elasticNet_pred = elasticNet_clf.predict(xtest)
elasticNet_ts_err = 1 - elasticNet_clf.score(xtest, ytest)
elasticNet_tr_err = 1 - elasticNet_clf.score(xtrain, ytrain)

# show result
print("elasticNet regression classification training error: %s \n testing error: %s \n"
    %(elasticNet_tr_err,elasticNet_ts_err))    

## ===========
# Compute regularization paths

# Ridge regression path
print("Computing regularization path using the ridge...")
n_alphas = 200
alphas = np.logspace(-4, 2, n_alphas)
clf = linear_model.Ridge(fit_intercept=False)

coefs = []
for a in alphas:
    clf.set_params(alpha=a, max_iter=1000)
    clf.fit(xtrain, ytrain)
    coefs.append(clf.coef_)

# lasso and elastic net path
eps = 5e-3  # the smaller it is the longer is the path

print("Computing regularization path using the lasso...")
alphas_lasso, coefs_lasso, _ = lasso_path(xtrain, ytrain, eps, fit_intercept=False)


print("Computing regularization path using the elastic net...")
alphas_enet, coefs_enet, _ = enet_path(
    xtrain, ytrain, eps=eps, l1_ratio=0.8, fit_intercept=False)

# Display results

plt.figure(1)
ax = plt.gca()
ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

plt.figure(2)
ax = plt.gca()
ax.set_color_cycle(2 * ['b', 'r', 'g', 'c', 'k'])
l1 = plt.plot(-np.log10(alphas_lasso), coefs_lasso.T)


plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Lasso coefficients as a function of the regularization')
plt.legend([l1[-1]], ['Lasso'], loc='lower left')
plt.axis('tight')


plt.figure(3)
ax = plt.gca()
ax.set_color_cycle(2 * ['b', 'r', 'g', 'c', 'k'])
l1 = plt.plot(-np.log10(alphas_enet), coefs_enet.T)


plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Elastic-Net coefficients as a function of the regularization')
plt.legend([l1[-1]], ['Elastic-Net'],
           loc='lower left')
plt.axis('tight')
plt.show()

    
