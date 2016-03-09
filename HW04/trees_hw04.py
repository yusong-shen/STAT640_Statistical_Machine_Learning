# -*- coding: utf-8 -*-
"""

Prediction using Trees

Created on Sun Nov 22 12:21:24 2015

@author: yusong
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import pydot # some parsing error

training_path = r'/Users/yusong/Code/STAT640/data/author_training.csv'
test_path = r'/Users/yusong/Code/STAT640/data/author_testing.csv'


# DataFrame
training = pd.read_csv(training_path)
test = pd.read_csv(test_path)


# convert dataframe to numpy array
# then use sklearn.cluster
words = training.columns.values.tolist()[:-1]
# remove the author labels
training_data = np.array(training.values[:,:-1])
training_authors = training.values[:,-1].tolist()
test_data = np.array(test.values[:,:-1])
test_authors = test.values[:,-1].tolist()
# use set()
author_names = ['Austen', 'London', 'Milton', 'Shakespeare']
label_map = {'Austen' : 0,
             'London' : 1,
             'Milton' : 2,
             'Shakespeare' : 3}
n_authors = len(label_map)
training_label = [label_map[i] for i in training_authors]
test_label = [label_map[i] for i in test_authors]


###############################################################################

# Classification tree
criterions = ['gini', 'entropy']
ctree_gini = DecisionTreeClassifier(random_state=0)
ctree_gini.fit(training_data, training_label)
err_ctree1_tr = ctree_gini.score(training_data, training_label)
err_ctree1_ts = ctree_gini.score(test_data,test_label)
#1
#0.876984126984
## Interpretation
export_graphviz(ctree_gini, out_file='ctree_gini.dot', 
                feature_names=words, class_names=author_names,
                filled=True, rounded=True,
                special_characters=True)
graph_gini = pydot.graph_from_dot_file('ctree_gini.dot')
graph_gini.write_png('ctree_gini.png')
# feature evaluation
# argsort return a increasing order
ind_gini = np.argsort(ctree_gini.feature_importances_)
# reverse the array to get decreasing order
features_gini = np.array(words)[ind_gini][::-1]

ctree = DecisionTreeClassifier(random_state=0, criterion='entropy')
ctree.fit(training_data, training_label)
err_ctree2_tr = ctree.score(training_data, training_label)
err_ctree2_tr = ctree.score(test_data,test_label)
#0.904761904762
export_graphviz(ctree, out_file='ctree_entropy.dot',
                feature_names=words, class_names=author_names,
                filled=True, rounded=True,
                special_characters=True)
graph_gini = pydot.graph_from_dot_file('ctree_entropy.dot')
graph_gini.write_png('ctree_entropy.png')
# feature evaluation
ind_entropy = np.argsort(ctree.feature_importances_)
features_entropy = np.array(words)[ind_entropy][::-1]

###############################################################################
# Bagging
bagging = BaggingClassifier()
bagging.fit(training_data, training_label)
err_bag_tr =  bagging.score(training_data, training_label)
err_bag_ts =  bagging.score(test_data,test_label)
#0.996604414261
#0.94444444444


###############################################################################
# Boosting
# AdaBoost
adaboost = AdaBoostClassifier()
adaboost.fit(training_data, training_label)
err_ada_tr =  adaboost.score(training_data, training_label)
err_ada_ts =  adaboost.score(test_data,test_label)
#0.9015280135823429
#0.8134920634920634
ind_adaboost = np.argsort(adaboost.feature_importances_)
features_adaboost = np.array(words)[ind_adaboost][::-1]


# GradientBoosting
gradientboost = GradientBoostingClassifier()
gradientboost.fit(training_data, training_label)
err_gradient_tr =  gradientboost.score(training_data, training_label)
err_gradient_ts =  gradientboost.score(test_data,test_label)
#1.0
#0.956349206349
ind_gradientboost = np.argsort(gradientboost.feature_importances_)
features_gradientboost = np.array(words)[ind_gradientboost][::-1]

###############################################################################
# Random Forests
rf = RandomForestClassifier()
rf.fit(training_data, training_label)
err_rf_tr =  rf.score(training_data, training_label)
err_rf_ts =  rf.score(test_data,test_label)
#0.998302207131
#0.968253968254
ind_rf = np.argsort(rf.feature_importances_)
features_rf = np.array(words)[ind_rf][::-1]

