# -*- coding: utf-8 -*-
"""
STAT640 HW03 
Problem 02 Digit data

Created on Mon Oct 26 16:03:32 2015

@author: yusong
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF, FastICA


tr_path = r'/Users/yusong/Code/STAT640/data/zip.train'
ts_path = r'/Users/yusong/Code/STAT640/data/zip.test'

# numpy ndarray
tr = np.genfromtxt(tr_path, delimiter=' ')
ts = np.genfromtxt(ts_path, delimiter=' ')

tr_feat = tr[:,1:]
ts_feat = ts[:,1:]
feat = np.concatenate((tr_feat, ts_feat))
tr_label = tr[:,0]
ts_label = ts[:,0]
label = np.append(tr_label, ts_label)

# seperate every digit
digit_mat_array = []
digit_ts_mat_array = []
for i in range(10):
    digit_mat = feat[label==i]
    digit_mat_array.append(digit_mat)
    digit_ts_mat = ts_feat[ts_label==i]
    digit_ts_mat_array.append(digit_ts_mat)

def cal_num_of_component(explained_variance_ratio, threshold):
    cum_var = 0
    num_component = 0
    for ind in range(len(explained_variance_ratio)):
        if cum_var <= threshold:
            cum_var += pca.explained_variance_ratio_[ind]
        else :
            num_component = ind + 1
            break    
    return num_component

# PCA
# keep all the component
for i in range(10):
    pca = PCA()
    pca.fit(digit_mat_array[i])
    print "pca - digit %s"%i
    print "percentage of variance by first 9 components:"
    print pca.explained_variance_ratio_[:9]
    print "cumulative sum:"
    print pca.explained_variance_ratio_[:9].cumsum()
    print "number of component that retain 95% variance:"
    num_component = cal_num_of_component(pca.explained_variance_ratio_, 0.95)
    print num_component
            
    # plot the figure        
    plt.figure()
    filename = "pca_digit_%s_components.png"%i
    for j in range(9):
        plt.subplot(3,3,j)
#        plt.gray()
        plt.imshow(pca.components_[j].reshape(16,16))
    plt.title("pca_digit_%s_components"%i, y=-0.5)
    plt.savefig(filename)    


# NMF
for i in range(10):
    nmf = NMF()
    nmf.fit(digit_mat_array[i]+1)
    plt.figure()    
    for j in range(9):
        plt.subplot(3,3,j)
#        plt.gray()
        plt.imshow(nmf.components_[j].reshape(16,16))
    plt.title("nmf_digit_%s_components"%i, y=-0.5)            
    filename = "nmf_digit_%s_components.png"%i
    plt.savefig(filename)    


# ICA
for i in range(10):
    ica = FastICA()
    ica.fit(digit_mat_array[i])
    plt.figure()    
    for j in range(9):
        plt.subplot(3,3,j)
#        plt.gray()
        plt.imshow(ica.components_[j].reshape(16,16))
    plt.title("ica_digit_%s_components"%i, y=-0.5)            
    filename = "ica_digit_%s_components.png"%i
    plt.savefig(filename)    

factor_array = [10, 20 ,50, 250]
for factor in factor_array:
    nmf = NMF(n_components=factor)
    nmf.fit(digit_mat_array[3]+1)
    plt.figure()    
    for j in range(9):
        plt.subplot(3,3,j)
    #        plt.gray()
        plt.imshow(nmf.components_[j].reshape(16,16))
    plt.title("nmf_digit3_factor%s_components"%factor, y=-0.5)            
    filename = "nmf_digit3_factor%s_components.png"%factor
    plt.savefig(filename)    

for factor in factor_array:
    ica = FastICA(n_components=factor)
    ica.fit(digit_mat_array[3])
    plt.figure()    
    for j in range(9):
        plt.subplot(3,3,j)
    #        plt.gray()
        plt.imshow(ica.components_[j].reshape(16,16))
    plt.title("ica_digit3_factor%s_components"%factor, y=-0.5)            
    filename = "ica_digit3_factor%s_components.png"%factor
    plt.savefig(filename)    
