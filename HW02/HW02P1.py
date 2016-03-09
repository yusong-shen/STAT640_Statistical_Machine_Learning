# -*- coding: utf-8 -*-
"""

HW02P1

Digits data: 3 and 8

Logistic regression without and with regularization

Created on Wed Oct 14 00:22:45 2015

@author: yusong
"""

import numpy as np
from sklearn import linear_model, metrics


tr_feat_path = r'/Users/yusong/Code/STAT640/HW02/zip38_train.csv'
ts_feat_path = r'/Users/yusong/Code/STAT640/HW02/zip38_test.csv'

tr_label_path = r'/Users/yusong/Code/STAT640/HW02/zip38_train_labels.csv'
ts_label_path = r'/Users/yusong/Code/STAT640/HW02/zip38_test_labels.csv'

# numpy ndarray
tr_feat = np.genfromtxt(tr_feat_path, delimiter=',')
ts_feat = np.genfromtxt(ts_feat_path, delimiter=',')
tr_label = np.genfromtxt(tr_label_path, delimiter=',')
ts_label = np.genfromtxt(ts_label_path, delimiter=',')


lg = linear_model.LogisticRegression(penalty='l2', C=1e8)
lg.fit(tr_feat, tr_label)

lg_preds = lg.predict(ts_feat)
lg_err = 1 - lg.score(ts_feat, ts_label)

lg_cmat = metrics.confusion_matrix(ts_label, lg_preds)


print("L2 Regularized Logistic Regression- classification err: %s \n"%(lg_err))
print("confusion matrix: \n %s"%(lg_cmat))

