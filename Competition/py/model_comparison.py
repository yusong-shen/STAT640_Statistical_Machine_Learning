# -*- coding: utf-8 -*-
"""
Model Comparison

Created on Mon Sep 28 09:23:50 2015

@author: yusong
"""

import graphlab as gl
#from scipy import sparse
#import numpy as np

rpath = r'/Users/yusong/Code/STAT640/Competition/data/trainRatings.csv'
idmap_path = r'/Users/yusong/Code/STAT640/Competition/data/testIDmap2Col.csv'
kaggleid_path = r'/Users/yusong/Code/STAT640/Competition/data/testKaggleID.csv'
testSet_path = r'/Users/yusong/Code/STAT640/Competition/data/testRatings.csv'
ratings = gl.SFrame.read_csv(rpath)
idmap = gl.SFrame.read_csv(idmap_path)
kaggleID = gl.SFrame.read_csv(kaggleid_path)
testRatings = gl.SFrame.read_csv(testSet_path)

m1 = gl.factorization_recommender.create(ratings,user_id="UserID",
                                item_id="ProfileID",target="Rating")
                                                                            
  

m2 = gl.item_similarity_recommender.create(ratings,user_id="UserID",
                                item_id="ProfileID",target="Rating")
                                

gl.recommender.util.compare_models(testRatings, [m1, m2, m3, m4],
                                   metric='rmse')