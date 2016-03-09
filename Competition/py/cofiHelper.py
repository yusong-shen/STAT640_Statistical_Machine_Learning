# -*- coding: utf-8 -*-
"""

cofi helper
Created on Thu Nov 26 22:57:49 2015

@author: yusong
"""
import numpy as np
from scipy.sparse import csr_matrix
import random

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def normalizeRatings(Y, R):
    """
    Preprocess data by subtracting mean rating for every 
    profile (every column)
    """
    m, n = Y.shape
    Ymean = np.zeros(m)
    Ynorm = np.zeros((m, n))
    for i in range(m):
        idx = np.where(R[i,:])
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]
    return Ynorm, Ymean

def cofiCost(params, Y, R, n_users, n_profiles, n_features, lambdaa):
    """
    Collaborative filtering cost function
    """
    # Unfold the U and W matrices from params
    X = np.reshape(params[:n_profiles*n_features], (n_profiles, n_features))
    Theta = np.reshape(params[n_profiles*n_features:], (n_users, n_features))

    se = (np.square(np.dot(X, Theta.T)-Y))
    J = 0.5*(np.sum(np.sum(np.multiply(se, R), axis=0), axis=0)) + \
        lambdaa/2.0*np.sum(np.sum(np.power(Theta, 2), axis=0), axis=0)+ \
        lambdaa/2.0*np.sum(np.sum(np.power(X, 2), axis=0), axis=0)
    return J
    
def cofiGrad(params, Y, R, n_users, n_profiles, n_features, lambdaa):
    """
    Collaborative filtering gradient function
    """
    X = np.reshape(params[:n_profiles*n_features], (n_profiles, n_features))
    Theta = np.reshape(params[n_profiles*n_features:], (n_users, n_features))

    X_grad = np.dot(np.multiply((np.dot(X, Theta.T)-Y), R), Theta) + \
        lambdaa * X
    Theta_grad = np.dot(np.multiply((np.dot(X, Theta.T)-Y), R).T, X) + \
        lambdaa * Theta
    grad = np.r_[X_grad.flatten(), Theta_grad.flatten()]
    return grad


def splitDataset(Y, R, test_size):
    """
    split the rating matrix
    """
    i, u = Y.shape
    n = np.sum(R!=0)
    n_test = int(n * test_size)
    n_train = n - n_test
    R = csr_matrix(R).nonzero()
    idx = range(n)
    random.shuffle(idx)
    train_idx = np.array(idx[:n_train])
    test_idx = np.array(idx[n_train:])
    
    R_train_idx = (R[0][train_idx],
               R[1][train_idx]) 
    R_train = csr_matrix((np.ones(n_train),R_train_idx),shape=(i,u))
    
    R_test_idx = (R[0][test_idx],
               R[1][test_idx])
    R_test = csr_matrix((np.ones(n_test),R_test_idx),shape=(i,u))

    Y_train = np.multiply(Y, R_train.toarray())
    Y_test = np.multiply(Y, R_test.toarray())
    return Y_train, Y_test, R_train, R_test

#    
#X = np.array([[1.0487,   -0.4002,    1.194],
#              [0.7809,   -0.3856,    0.521],
#              [0.6415,   -0.5479,   -0.083],
#              [0.4536,   -0.8002,    0.680],
#              [0.9375,    0.1061,    0.362]])
#              
#Y = np.array([[5,     4,     0,     0],
#              [3,     0,     0,     0],
#              [4,     0,     0,     0],
#              [3,     0,     0,     0],
#              [3,     0,     0,     0]])
#
#Theta = np.array([[0.2854,   -1.6843,    0.2629],
#              [0.5050,   -0.4546,    0.3175],
#              [-0.4319,   -0.4788,    0.8467],
#              [0.7286,   -0.2719,    0.3268]])
#
#R = (Y!=0)
#params = np.r_[X.flatten(), Theta.flatten()]
#lambdaa = 1.5
#n_users = 4
#n_profiles = 5
#n_features = 3
#J = cofiCost(params, Y, R, n_users, n_profiles, n_features, lambdaa)
#grad = cofiGrad(params, Y, R, n_users, n_profiles, n_features, lambdaa)
#
#test_size = 0.2
#Y_train, Y_test, R_train, R_test = splitDataset(Y, R, test_size)
#
#Ynorm_train, Ymean_train = normalizeRatings(Y, R);
