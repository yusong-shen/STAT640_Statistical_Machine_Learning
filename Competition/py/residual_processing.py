# -*- coding: utf-8 -*-
"""

STAT640 Competition

Residual processing


Created on Sun Nov  1 20:20:24 2015

@author: yusong
"""

import graphlab as gl
import graphlab.numpy
import numpy as np
from helper_functions import modify_rating, discrete_ratings
import matplotlib.pyplot as plt
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
train_idmap = training[['UserID', 'ProfileID']]
val_idmap = validation[['UserID', 'ProfileID']]
test_idmap = testing[['UserID', 'ProfileID']]

optimal_params = {'linear_regularization': 1e-05,
 'max_iterations': 50,
 'num_factors': 5,
 'regularization': 1e-06}

 
mf = gl.factorization_recommender.create(training,user_id="UserID",
                                item_id="ProfileID",target="Rating",
                                **optimal_params)

# training error
pred_train = mf.predict(train_idmap)
train_rmse = mf.evaluate_rmse(training, target="Rating")
train_rmse_np = gl.numpy.array(train_rmse)
print "training rmse is : %s"%train_rmse['rmse_overall']
# training rmse is : 1.62812638743


# validation error                                                                            
pred_val = mf.predict(val_idmap)
val_rmse = mf.evaluate_rmse(validation, target="Rating")
val_rmse_np = gl.numpy.array(val_rmse)
print "validation rmse is : %s"%val_rmse['rmse_overall']
# validation rmse is : 1.74713030669


# calculate the residual
train_residual = training['Rating'] - pred_train
val_residual = validation['Rating'] - pred_val
# visualize residual
hist_val, bins_val = np.histogram(val_residual, bins=50)
hist_tr, bins_tr = np.histogram(train_residual, bins=50)
fig = plt.figure()

plt.subplot(2,1,1)
width = 0.9 * (bins_tr[1] - bins_tr[0])
center = (bins_tr[:-1] + bins_tr[1:]) / 2
plt.bar(center, hist_tr, align='center', width=width)
plt.title("MF residual in training set")

plt.subplot(2,1,2)
width = 0.9 * (bins_val[1] - bins_val[0])
center = (bins_val[:-1] + bins_val[1:]) / 2
plt.bar(center, hist_val, align='center', width=width)
plt.title("MF residual in validation set")
plt.show()


fig_path = r'/Users/yusong/Code/STAT640/Competition/data/output'
fig.savefig(fig_path+r'/residual_hist.png')

##  ====== stage 2
## use item similarity model to learn the residual
tr_residuals_mat = gl.SFrame()
tr_residuals_mat.add_columns(train_idmap)
tr_residuals_mat.add_column(train_residual, name='Residual')

val_residuals_mat = gl.SFrame()
val_residuals_mat.add_columns(val_idmap)
val_residuals_mat.add_column(val_residual, name='Residual')

m_sim = gl.item_similarity_recommender.create(tr_residuals_mat,user_id="UserID",
                                item_id="ProfileID",target="Residual")

pred_re_tr = m_sim.predict(train_idmap)                            
re_tr_rmse = m_sim.evaluate_rmse(tr_residuals_mat, target='Residual')
print "residual prediction training error :%s"%re_tr_rmse['rmse_overall']
#residual prediction training error :1.66226504394


pred_re_val = m_sim.predict(val_idmap)
re_val_rmse = m_sim.evaluate_rmse(val_residuals_mat, target='Residual')
print "residual prediction validation error :%s"%re_val_rmse['rmse_overall']
#residual prediction validation error :1.77796415199

# add the predicting residual back to mf prediction

pred_tr_final = pred_train + pred_re_tr
pred_val_final = pred_val + pred_re_val

final_tr_rmse = gl.evaluation.rmse(training["Rating"], pred_tr_final)
final_val_rmse = gl.evaluation.rmse(validation["Rating"], pred_val_final)

print "final training rmse is : %s"%final_tr_rmse
print "final validation rmse is : %s"%final_val_rmse
# even worse
#final training rmse is : 1.66226504394
#final validation rmse is : 1.77796415199

## use popularity model to learn the residual
m_pop = gl.popularity_recommender.create(tr_residuals_mat,user_id="UserID",
                                item_id="ProfileID",target="Residual")

pred_re_tr = m_pop.predict(train_idmap)                            
re_tr_rmse = m_pop.evaluate_rmse(tr_residuals_mat, target='Residual')
print "residual prediction training error :%s"%re_tr_rmse['rmse_overall']


pred_re_val = m_pop.predict(val_idmap)
re_val_rmse = m_pop.evaluate_rmse(val_residuals_mat, target='Residual')
print "residual prediction validation error :%s"%re_val_rmse['rmse_overall']

# add the predicting residual back to mf prediction

pred_tr_final = pred_train + pred_re_tr
pred_val_final = pred_val + pred_re_val

final_tr_rmse = gl.evaluation.rmse(training["Rating"], pred_tr_final)
final_val_rmse = gl.evaluation.rmse(validation["Rating"], pred_val_final)

print "final training rmse is : %s"%final_tr_rmse
print "final validation rmse is : %s"%final_val_rmse
#final training rmse is : 1.62661117911
#final validation rmse is : 1.74570003611
# slightly better

## ============ stage 3
## final output
mf = gl.factorization_recommender.create(training_whole,user_id="UserID",
                                item_id="ProfileID",target="Rating",
                                **optimal_params)
# Final training RMSE: 1.6355
pred_train = mf.predict(train_idmap)
pred_test = mf.predict(test_idmap)

   
# Calculate the residual of whole training set                            
whole_residual = training['Rating'] - pred_train                              
whole_residuals_mat = gl.SFrame()
whole_residuals_mat.add_columns(train_idmap)
whole_residuals_mat.add_column(whole_residual, name='Residual')

# predict the residual used popularity model
m_pop = gl.popularity_recommender.create(whole_residuals_mat,user_id="UserID",
                                item_id="ProfileID",target="Residual")

# predict the residual for test set
pred_re_test = m_pop.predict(test_idmap)

pred_test_final = pred_test + pred_re_test
final_test_rmse = gl.evaluation.rmse(testing["Rating"], pred_test_final)

print "final test rmse is : %s"%final_test_rmse

pred_test_final = discrete_ratings(pred_test_final,threshold=0.5)
final_test_rmse = gl.evaluation.rmse(testing["Rating"], pred_test_final)
print "final test rmse after modified is : %s"%final_test_rmse


# ouput for test
pred_comp = mf.predict(idmap)
pred_re_comp = m_pop.predict(idmap)
pred_comp_final = pred_comp + pred_re_comp
pred_comp_final = modify_rating(pred_comp_final)
pred = gl.SFrame.add_column(kaggleID, pred_comp_final)
pred.rename({'X2':'Prediction'})
pred_path = r'/Users/yusong/Code/STAT640/Competition/data/output/residual_processing.csv'
pred.export_csv(pred_path)
