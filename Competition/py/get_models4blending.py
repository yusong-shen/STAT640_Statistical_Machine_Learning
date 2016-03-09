# -*- coding: utf-8 -*-
"""
Get multiple models for blending

Created on Mon Nov 30 12:50:06 2015

@author: yusong
"""

import graphlab as gl
#from scipy import sparse
#import numpy as np
import scipy.io as sio
import graphlab.numpy as gnp

train_path = r'/Users/yusong/Code/STAT640/Competition/data/trainRatings.csv'
probe_train_path = r'/Users/yusong/Code/STAT640/Competition/data/probe_train.csv'
probe_validate_path = r'/Users/yusong/Code/STAT640/Competition/data/probe_validate.csv'
gender_path = r'/Users/yusong/Code/STAT640/Competition/data/gender.csv'
idmap_path = r'/Users/yusong/Code/STAT640/Competition/data/IDMap.csv'


train = gl.SFrame.read_csv(train_path)
probe_train = gl.SFrame.read_csv(probe_train_path)
probe_validate = gl.SFrame.read_csv(probe_validate_path)
gender = gl.SFrame.read_csv(gender_path)
idmap = gl.SFrame.read_csv(idmap_path)

# train the factorization recommemder with gender info
Ds = [5,10]
for D in Ds:
    m1 = gl.factorization_recommender.create(train,user_id="UserID",
                                    item_id="ProfileID",target="Rating",
                                    user_data=gender,
                                    num_factors=D,)
    filename = r'/Users/yusong/Code/STAT640/Competition/blending/models/fm_D%s.mat'%(D)
                                                                                
      
    
    # predict
    train_idmap = train.select_columns(["UserID", "ProfileID"])
    pred_train = gnp.array(m1.predict(train_idmap))
    probe_train_idmap = probe_train.select_columns(["UserID", "ProfileID"])
    pred_probe = gnp.array(m1.predict(probe_train_idmap))
    rmse = m1.evaluate_rmse(probe_train, target='Rating')['rmse_overall']
    probe_validate_idmap = probe_validate.select_columns(["UserID", "ProfileID"])
    pred_validate = gnp.array(m1.predict(probe_validate_idmap))
    kaggle_idmap = idmap.select_columns(["UserID", "ProfileID"])
    pred_kaggle = gnp.array(m1.predict(kaggle_idmap))
    
    ## write to .mat
    sio.savemat(filename,{"pred_train":pred_train,
                          "pred_probe":pred_probe,
                          "pred_test":pred_validate,
                          "rmse":rmse,
                          "pred_kaggle":pred_kaggle} )
    print "D : %s, rmse : %s"%(D, rmse)
    


# different knn
n_item = 10000
ks = [5, 100, 200, n_item]
similarities = ['jaccard', 'cosine', 'pearson']
for K in ks:
    for sim in similarities:
        m1 = gl.item_similarity_recommender.create(train,user_id="UserID",
                                        item_id="ProfileID",target="Rating",
                                        only_top_k = K,
                                        similarity_type = sim)
        filename = r'/Users/yusong/Code/STAT640/Competition/blending/models/knn_K%s_Sim_%s.mat'%(K,sim)
                                                                                    
        # predict
        train_idmap = train.select_columns(["UserID", "ProfileID"])
        pred_train = gnp.array(m1.predict(train_idmap))
        probe_train_idmap = probe_train.select_columns(["UserID", "ProfileID"])
        pred_probe = gnp.array(m1.predict(probe_train_idmap))
        rmse = m1.evaluate_rmse(probe_train, target='Rating')['rmse_overall']
        probe_validate_idmap = probe_validate.select_columns(["UserID", "ProfileID"])
        pred_validate = gnp.array(m1.predict(probe_validate_idmap))
        kaggle_idmap = idmap.select_columns(["UserID", "ProfileID"])
        pred_kaggle = gnp.array(m1.predict(kaggle_idmap))
        
        ## write to .mat
        sio.savemat(filename,{"pred_train":pred_train,
                              "pred_probe":pred_probe,
                              "pred_test":pred_validate,
                              "rmse":rmse,
                              "pred_kaggle":pred_kaggle} )
        print "k : %s, rmse : %s"%(K, rmse)
        
