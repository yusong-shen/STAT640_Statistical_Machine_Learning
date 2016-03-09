# -*- coding: utf-8 -*-
"""

STAT640 Competition

Residual processing


Created on Sun Nov  1 20:20:24 2015

@author: yusong
"""

import graphlab as gl
import numpy as np
from helper_functions import modify_rating
#from scipy import sparse
#import numpy as np

rpath = r'/Users/yusong/Code/STAT640/Competition/data/ratings.csv'
idmap_path = r'/Users/yusong/Code/STAT640/Competition/data/IDmap2Col.csv'
kaggleid_path = r'/Users/yusong/Code/STAT640/Competition/data/KaggleID.csv'
ratings = gl.SFrame.read_csv(rpath)
idmap = gl.SFrame.read_csv(idmap_path)
kaggleID = gl.SFrame.read_csv(kaggleid_path)


# Split the data set
training_whole, testing = ratings.random_split(0.9, seed=1)
training, validation = training_whole.random_split(0.9, seed=1)

optimal_params = {'linear_regularization': 1e-05,
 'max_iterations': 50,
 'num_factors': 5,
 'regularization': 1e-06}

 
m1 = gl.factorization_recommender.create(training,user_id="UserID",
                                item_id="ProfileID",target="Rating",
                                **optimal_params)
                                                                            
pred


