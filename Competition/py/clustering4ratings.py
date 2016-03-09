# -*- coding: utf-8 -*-
"""

Clustering for online dating ratings feature matrix

Created on Tue Dec  1 11:55:22 2015

@author: yusong
"""

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#import h5py
import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering

gender_path = r'/Users/yusong/Code/STAT640/Competition/data/gender.csv'


## reading the feature matrix U, F
matpath = r'/Users/yusong/Code/STAT640/Competition/export_bptf4competition/output/pmf200.mat'

#bptf_mat = h5py.File(matpath)
#bptf_mat.keys()
#
#U = bptf_mat['Us_bptf']

pmf_mat = sio.loadmat(matpath)
# Factors * #users
U = pmf_mat['U']
V = pmf_mat['V']

# 1 : Female 2 : Male 3 : Unknown
gender = pd.read_csv(gender_path).values[:,1]
n_gender = 3


## use PCA and LDA to visualize data
X = U.T
y = np.array(gender)
target_names = ['Female', 'Male', 'Unknown']

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

fig = plt.figure()
for c, i, target_name in zip("rgb", range(1,n_gender+1), target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
plt.title('PCA of PMF U matrix', y=1.05)
#fig.savefig('PCA_pmf.png')

fig = plt.figure()
for c, i, target_name in zip("rgb", range(1,n_gender+1), target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], c=c, label=target_name)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
plt.title('LDA of PMF U matrix', y=1.05)
#fig.savefig('LDA_pmf.png')

plt.show()

##
# use kmeans clustering
# Note : still fit to unreduced data
n_clusters = 2
target_names = ["Cluster %s"%i for i in range(2) ]
km = KMeans(n_clusters = n_clusters)
km.fit(X)
y_pred = np.array(km.labels_)
fig = plt.figure()
for c, i, target_name in zip("rg", range(n_clusters), target_names):
    plt.scatter(X_r[y_pred == i, 0], X_r[y_pred == i, 1], c=c, label=target_name)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
plt.title('Kmeans clustering in PCA reduced user feature matrix', y=1.05)
#fig.savefig('Kmeans_PCA_authorship.png')

fig = plt.figure()
for c, i, target_name in zip("rg", range(n_clusters), target_names):
    plt.scatter(X_r2[y_pred == i, 0], X_r2[y_pred == i, 1], c=c, label=target_name)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
plt.title('Kmeans clustering in LDA reduced user feature matrix', y=1.05)
#fig.savefig('Kmeans_LDA_authorship.png')

plt.show()


## 
# visualize Profile matrix
## use PCA and LDA to visualize data
X = V.T
y = np.array(gender)

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)


# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

fig = plt.figure()
plt.scatter(X_r[:, 0], X_r[:, 1])
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
plt.title('PCA of PMF V matrix', y=1.05)
#fig.savefig('PCA_pmf.png')


plt.show()

n_clusters = 2
target_names = ["Cluster %s"%i for i in range(2) ]
km = KMeans(n_clusters = n_clusters)
km.fit(X)
y_pred = np.array(km.labels_)
fig = plt.figure()
for c, i, target_name in zip("rg", range(n_clusters), target_names):
    plt.scatter(X_r[y_pred == i, 0], X_r[y_pred == i, 1], c=c, label=target_name)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
plt.title('Kmeans clustering in PCA reduced profile feature matrix', y=1.05)


## Hierachical Clustering
linkages = ["complete", "average"]
affinities = ["euclidean", "l1", "l2", "cosine"]
def hierarchical_clustering(data, vlinkage, vaffinity, n_clusters=2):
    """
    perform hierarchical clustering and plot
    """
    hc = AgglomerativeClustering(n_clusters=n_clusters,linkage=vlinkage, affinity=vaffinity)
    hc.fit(data)
    
    target_names = ["Cluster %s"%i for i in range(n_clusters) ]
    y_pred = np.array(hc.labels_)
    fig = plt.figure()
    for c, i, target_name in zip("rb", range(n_clusters), target_names):
        plt.scatter(X_r[y_pred == i, 0], X_r[y_pred == i, 1], c=c, label=target_name)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=3, fancybox=True, shadow=True)
    plt.title('Hierarchical clustering with %s and %s in PCA reduced V matrix'%(vlinkage, vaffinity),
              y=1.05)
#    fig.savefig('HC_%s_%s_PCA_authorship.png'%(vlinkage, vaffinity))

hierarchical_clustering(X, "ward", "euclidean")
for linkage in linkages:
    for affinity in affinities:
        hierarchical_clustering(X, linkage, affinity)



