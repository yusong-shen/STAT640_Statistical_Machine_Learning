# -*- coding: utf-8 -*-
"""

STAT640 HW02 P2
Digit Data

Created on Tue Oct 13 23:16:35 2015

Reference : http://scikit-learn.org/stable/modules/multiclass.html

@author: yusong
"""

import numpy as np
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

tr_path = r'/Users/yusong/Code/STAT640/HW02/zip.train'
ts_path = r'/Users/yusong/Code/STAT640/HW02/zip.test'

#tr_label_path = r'/Users/yusong/Code/STAT640/HW02/zip38_train_labels.csv'
#ts_label_path = r'/Users/yusong/Code/STAT640/HW02/zip38_test_labels.csv'

# numpy ndarray
tr = np.genfromtxt(tr_path, delimiter=' ')
ts = np.genfromtxt(ts_path, delimiter=' ')

tr_feat = tr[:,1:]
ts_feat = ts[:,1:]
tr_label = tr[:,0]
ts_label = ts[:,0]

# use sklearn C-Support Vector Classification
## == one-vs-one == ##
# The multiclass support is handled in a one-vs-one scheme
# train 
ovo_clf = OneVsOneClassifier(LinearSVC())
ovo_clf.fit(tr_feat, tr_label)

# predict
ovo_pred = ovo_clf.predict(ts_feat)
ovo_err = 1- ovo_clf.score(ts_feat, ts_label)

# confusion matrix
#
#array([[159,   7],
#       [  5, 161]])
ovo_cmat = metrics.confusion_matrix(ts_label, ovo_pred) 
pred_total = np.sum(ovo_cmat,axis = 1)
ovo_mis = 1- np.diag(ovo_cmat).astype(float) / pred_total
print("one vs. one svm - classification err: %s \n"%(ovo_err))
print("confusion matrix: \n %s"%(ovo_cmat))
print("class misclassification rate : \n %s"%(ovo_mis))
## == one-vs-rest == ##
# The multiclass support is handled in a one-vs-rest scheme
# train 
ovr_clf = OneVsRestClassifier(LinearSVC())
ovr_clf.fit(tr_feat, tr_label)

# predict
ovr_pred = ovr_clf.predict(ts_feat)
ovr_err = 1- ovr_clf.score(ts_feat, ts_label)

# confusion matrix
#
#array([[159,   7],
#       [  5, 161]])
ovr_cmat = metrics.confusion_matrix(ts_label, ovr_pred) 
ovr_mis = 1- np.diag(ovr_cmat).astype(float) / pred_total
print("one vs. rest svm - classification err: %s \n"%(ovr_err))
print("confusion matrix: \n %s"%(ovr_cmat))
print("class misclassification rate : \n %s"%(ovr_mis))


