# -*- coding: utf-8 -*-
"""
Stat640 Competition
Progression stage 2 - blending

Created on Fri Oct 30 16:08:02 2015

@author: yusong
"""

import graphlab as gl
import numpy as np
import graphlab.numpy
from helper_functions import modify_rating, discrete_ratings
#from scipy import sparse
#import numpy as np

rpath = r'/Users/yusong/Code/STAT640/Competition/data/ratings.csv'
idmap_path = r'/Users/yusong/Code/STAT640/Competition/data/IDmap2Col.csv'
kaggleid_path = r'/Users/yusong/Code/STAT640/Competition/data/KaggleID.csv'
ratings = gl.SFrame.read_csv(rpath)
idmap = gl.SFrame.read_csv(idmap_path)
kaggleID = gl.SFrame.read_csv(kaggleid_path)

## Split the data set
#training_rating, validation_rating = ratings.random_split(0.8, seed=1)
#                                                                            
optimal_params = {'linear_regularization': 1e-05,
 'max_iterations': 50,
 'num_factors': 5,
 'regularization': 1e-06}
 
m1 = gl.factorization_recommender.create(ratings,user_id="UserID",
                                item_id="ProfileID",target="Rating",
                                **optimal_params)
                                                                            
  

m2 = gl.item_similarity_recommender.create(ratings,user_id="UserID",
                                item_id="ProfileID",target="Rating")
                                
m3 = gl.popularity_recommender.create(ratings,user_id="UserID",
                                item_id="ProfileID",target="Rating") 
                                
# # the predition rating is negative ?
#m4 = gl.ranking_factorization_recommender.create(ratings,user_id="UserID",
#                                item_id="ProfileID",target="Rating")

def modify_rating(ratings):
    """
    make sure the rating range from 0 - 10,
    assume ratings is a SArray
    """
#    for i in range(ratings.size()):
#        if ratings[i] < 0:
#            ratings[i] = 0
#        elif ratings[i] > 10:
#            ratings[i] = 10
    rs = gl.numpy.array(ratings)
    rs[rs<0.5] = 0
    rs[rs>9.5] = 10
    return gl.SArray(rs)
            

    

# uniform blending
# query_result : graphlab.data_structures.sarray.SArray
query_result1 = m1.predict(idmap)
#query_result2 = m2.predict(idmap)
#query_result3 = m3.predict(idmap)
#query_result4 = m4.predict(idmap)
print query_result1[:10]
#print query_result2[:10]
#print query_result3[:10]

#query_result = (1.0/3)*(query_result1+query_result2+query_result3)
query_result = discrete_ratings(query_result1, threshold=0.5)
print query_result[:10]
pred = gl.SFrame.add_column(kaggleID, query_result)
pred.rename({'X2':'Prediction'})
pred_path = r'/Users/yusong/Code/STAT640/Competition/data/output/trunkingPlus_factorization_recommender.csv'
pred.export_csv(pred_path)

