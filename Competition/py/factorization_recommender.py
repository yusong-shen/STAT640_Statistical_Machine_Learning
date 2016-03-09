# -*- coding: utf-8 -*-
"""
STAT640 Competition
python script for Factorization Recommender


Created on Sun Sep 27 23:01:20 2015

@author: yusong
"""

import graphlab as gl
#from scipy import sparse
#import numpy as np

rpath = r'/Users/yusong/Code/STAT640/Competition/data/ratings.csv'
idmap_path = r'/Users/yusong/Code/STAT640/Competition/data/IDmap2Col.csv'
kaggleid_path = r'/Users/yusong/Code/STAT640/Competition/data/KaggleID.csv'
#testSet_path = r'/Users/yusong/Code/STAT640/Competition/data/testRatings.csv'
gender_path = r'/Users/yusong/Code/STAT640/Competition/data/gender.csv'

ratings = gl.SFrame.read_csv(rpath)
idmap = gl.SFrame.read_csv(idmap_path)
kaggleID = gl.SFrame.read_csv(kaggleid_path)
#testRatings = gl.SFrame.read_csv(testSet_path)
gender = gl.SFrame.read_csv(gender_path)

# Split the data set
#training, validation = ratings.random_split(0.8, seed=1)
#                                                                            
# Todo : tune the parameter
#params = {'target': 'Rating',
#          'user_id': 'UserID',
#          'item_id': 'ProfileID',
#          'num_factors': range(5,12)}
#job = gl.model_parameter_search.create((training, validation),
#                                gl.factorization_recommender.create,
#                                params,
#                                perform_trial_run=True)
#
#tune_results = job.get_results()
#tune_results.column_names()
#
#sorted_summary = tune_results.sort('validation_rmse', ascending=True)
#print sorted_summary
#
#optimal_model_idx = sorted_summary[0]['model_id']
#
#optimal_params = sorted_summary['linear_regularization', 'max_iterations',
#                                'num_factors', 'regularization'][0]
#optimal_rmse = sorted_summary[0]['validation_rmse']

optimal_params = {'linear_regularization': 1e-05,
 'max_iterations': 50,
 'num_factors': 5,
 'regularization': 1e-06}
# Use the optimal parameter to train the model
## Todo : use the side information : user's gender
#model = gl.factorization_recommender.create(ratings,user_id="UserID",
#                                item_id="ProfileID",target="Rating",
#                                user_data=gender,
#                                **optimal_params)
 
# Without gender information 
model = gl.factorization_recommender.create(ratings,user_id="UserID",
                                item_id="ProfileID",target="Rating",
                                **optimal_params)


query_result = model.predict(idmap)
pred = gl.SFrame.add_column(kaggleID, query_result)
pred.rename({'X2':'Prediction'})
pred_path = r'/Users/yusong/Code/STAT640/Competition/data/factorization_recommender_tune1.csv'
pred.export_csv(pred_path)


