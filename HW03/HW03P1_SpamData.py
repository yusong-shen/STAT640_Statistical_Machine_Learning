# -*- coding: utf-8 -*-
"""
STAT640 HW03
Problem 01 Spam data

Created on Mon Oct 26 16:02:18 2015

@author: yusong
"""

import numpy as np
#import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import hinge_loss, classification_report

data_path = r'/Users/yusong/Code/STAT640/data/spam.data'
label_path = r'/Users/yusong/Code/STAT640/data/spam.traintest'

# numpy ndarray
data = np.genfromtxt(data_path, delimiter=' ')
label = np.genfromtxt(label_path, delimiter=' ')

# Todo : kfold() function
data = data[:,:-1]
n = data.shape[0]
#kf = cross_validation.KFold(n, n_folds = 5)
#for train_ind, test_ind in kf:
#    X_train, X_test = data[train_ind], data[test_ind]
#    y_train, y_test = label[train_ind], data[test_ind]
#    
    
#    print X_train.shape, X_test.shape
#    
#
#####
## K-Fold Cross Validation
#param_grid = [{"C":[0.1, 1, 10, 100, 1000]}]    
## K = 5
#clf = GridSearchCV(LinearSVC(), param_grid, cv=5,
#                   scoring="precision_weighted")
#clf.fit(data, label)                   
#print("Best parameters set found on development set:")
#print 
#print(clf.best_params_)
#print 
#print("Grid scores on development set:")
#print
#for params, mean_score, scores in clf.grid_scores_:
#    print("%0.3f (+/-%0.03f) for %r"
#          % (mean_score, scores.std() * 2, params))
#print
#
## K = 10
#clf = GridSearchCV(LinearSVC(), param_grid, cv=10,
#                   scoring="precision_weighted")
#clf.fit(data, label)                   
#print("Best parameters set found on development set:")
#print 
#print(clf.best_params_)
#print 
#print("Grid scores on development set:")
#print
#for params, mean_score, scores in clf.grid_scores_:
#    print("%0.3f (+/-%0.03f) for %r"
#          % (mean_score, scores.std() * 2, params))
#print
#


### =================
# Process of Statistical Learning
# split the dataset 
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    data, label, test_size = 0.4, random_state=0)
print X_train.shape
# set the alternative parameter list     
params = [
    {'C':[0.1, 1, 10, 100], 'kernel':['linear']},
    {'C':[0.1, 1, 10, 100], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]},
    {'C':[0.1, 1, 10, 100], 'kernel':['poly'], 'degree':[2, 3, 4]},
    ]

print params
# K = 5
clf = GridSearchCV(svm.SVC(), params, cv=5,
                   scoring="precision_weighted")
clf.fit(X_train, y_train)                                    
print("Best parameters set found on development set:")
print 
print(clf.best_params_)
print 
print("Grid scores on development set:")
print
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
          % (mean_score, scores.std() * 2, params))
print

# use the best parameter to train the entire training set
y_pred = clf.predict(X_test)

print("Detailed classification report:")
print
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print
print(classification_report(y_test, y_pred))
print

with open("Output.txt", "w") as text_file:
    text_file.write("Best parameters set found on development set:")
    text_file.write("") 
    text_file.write(str(clf.best_params_))
    text_file.write("") 
    text_file.write("Grid scores on development set:")
    text_file.write("")
    for params, mean_score, scores in clf.grid_scores_:
        text_file.write("%0.3f (+/-%0.03f) for %r\n"
              % (mean_score, scores.std() * 2, params))
    text_file.write("")
    text_file.write("Detailed classification report:")
    text_file.write("")
    text_file.write("The model is trained on the full development set.")
    text_file.write("The scores are computed on the full evaluation set.")
    text_file.write("")
    text_file.write(classification_report(y_test, y_pred))
    text_file.write("")
