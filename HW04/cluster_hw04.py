# -*- coding: utf-8 -*-
"""
STAT640 Homework 4

Problem 1 : Visualization & Clustering
Data :
author_training.csv
author_testing.csv
4 Authors : {'Austen', 'London', 'Milton', 'Shakespeare'}
n = 841, p = 69

Goal : 
use the word counts to cluster the data and see if your groups
coincide with the true author attribution

Created on Thu Nov 19 15:13:56 2015

@author: yusong
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from copy import copy
from scipy.cluster.hierarchy import dendrogram, linkage

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
training_label = training.values[:,-1].tolist()
test_data = np.array(test.values[:,:-1])
test_label = test.values[:,-1].tolist()
data = np.concatenate((training_data, test_data))
authors = training_label + test_label
# use set()
label_map = {'Austen' : 0,
             'London' : 1,
             'Milton' : 2,
             'Shakespeare' : 3}
label = [label_map[i] for i in authors]
n_authors = len(label_map)


###############################################################################
# How to visualize in low dimension ?
# Visualize in PCA-reduced data


## use PCA and LDA to visualize data
X = data
y = np.array(label)
target_names = ['Austen', 'London', 'Milton', 'Shakespeare']

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

fig = plt.figure()
for c, i, target_name in zip("rgbk", range(n_authors), target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
plt.title('PCA of authorship dataset', y=1.05)
fig.savefig('PCA_authorship.png')

fig = plt.figure()
for c, i, target_name in zip("rgbk", range(n_authors), target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], c=c, label=target_name)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
plt.title('LDA of authorship dataset', y=1.05)
fig.savefig('LDA_authorship.png')

plt.show()

###############################################################################
# use kmeans clustering
# Note : still fit to unreduced data
target_names = ["Cluster %s"%i for i in range(4) ]
km = KMeans(n_clusters = n_authors)
km.fit(data)
y_pred = np.array(km.labels_)
fig = plt.figure()
for c, i, target_name in zip("rgbk", range(n_authors), target_names):
    plt.scatter(X_r[y_pred == i, 0], X_r[y_pred == i, 1], c=c, label=target_name)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
plt.title('Kmeans clustering in PCA reduced authorship dataset', y=1.05)
fig.savefig('Kmeans_PCA_authorship.png')

fig = plt.figure()
for c, i, target_name in zip("rgbk", range(n_authors), target_names):
    plt.scatter(X_r2[y_pred == i, 0], X_r2[y_pred == i, 1], c=c, label=target_name)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
plt.title('Kmeans clustering in LDA reduced authorship dataset', y=1.05)
fig.savefig('Kmeans_LDA_authorship.png')

plt.show()

def cluster_accuracy(match, estimator, label):
    # Match has to be obeserved by eyes...
    # has to use shallow copy
    y_pred = np.array(estimator.labels_)
    y_pred_copy = copy(y_pred)
    for i in match.keys():
        y_pred[y_pred_copy==i] = match[i]
    acc = metrics.accuracy_score(y_pred, label)
    return acc
    
# How to know which cluster correspond to which author?
# Obeserve by eyes...
match = {0:1, 1:3, 2:2, 3:0}
print cluster_accuracy(match, km, label)
#0.77170035671

###############################################################################
# use Hierarchical Clustering
linkages = ["complete", "average"]
affinities = ["euclidean", "l1", "l2", "cosine"]
def hierarchical_clustering(data, vlinkage, vaffinity, n_clusters=4):
    """
    perform hierarchical clustering and plot
    """
    hc = AgglomerativeClustering(n_clusters=n_clusters,linkage=vlinkage, affinity=vaffinity)
    hc.fit(data)
    
    target_names = ["Cluster %s"%i for i in range(4) ]
    y_pred = np.array(hc.labels_)
    fig = plt.figure()
    for c, i, target_name in zip("rgbk", range(n_authors), target_names):
        plt.scatter(X_r[y_pred == i, 0], X_r[y_pred == i, 1], c=c, label=target_name)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=3, fancybox=True, shadow=True)
    plt.title('Hierarchical clustering with %s and %s in PCA reduced authorship dataset'%(vlinkage, vaffinity),
              y=1.05)
    fig.savefig('HC_%s_%s_PCA_authorship.png'%(vlinkage, vaffinity))
    
    fig = plt.figure()
    for c, i, target_name in zip("rgbk", range(n_authors), target_names):
        plt.scatter(X_r2[y_pred == i, 0], X_r2[y_pred == i, 1], c=c, label=target_name)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=3, fancybox=True, shadow=True)
    plt.title('Hierarchical clustering with %s and %s in LDA reduced authorship dataset'%(vlinkage, vaffinity), 
              y=1.05)
    fig.savefig('HC_%s_%s_LDA_authorship.png'%(vlinkage, vaffinity))
    
    plt.show()
    return hc

#hc_ward = hierarchical_clustering(data, "ward", "euclidean")
#for linkage in linkages:
#    for affinity in affinities:
#        hierarchical_clustering(data, linkage, affinity)
#
#
#
#match = {0:3, 1:1, 2:0, 3:2}
#print cluster_accuracy(match, hc_ward, label)
##0.780023781213

###############################################################################
## Draw  dendrogram
Z = linkage(data, 'ward')
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
    labels=np.array(authors)
)
plt.show()
 

###############################################################################
## Biclustering
data = data.astype('float')
bc = SpectralBiclustering(n_clusters=(n_authors,5))
bc.fit(data)
## TODO : sort the rows and columns 
bc_data = data[np.argsort(bc.row_labels_)]
bc_data = bc_data[:, np.argsort(bc.column_labels_)]
## How to annotate the words?
plt.matshow(data, cmap = plt.cm.Blues)
plt.title("Original dataset")
plt.matshow(bc_data, cmap = plt.cm.Blues)
plt.title("After biclustering; rearrange to show biclusters")