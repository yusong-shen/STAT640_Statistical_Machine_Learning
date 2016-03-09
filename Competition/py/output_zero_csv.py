# -*- coding: utf-8 -*-
"""
output competition required csv

Created on Mon Nov 23 13:18:26 2015

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

## ouput for test
#length = kaggleID.shape[0]
#zero_pred = gl.SArray([0 for i in range(length)])
#pred = gl.SFrame.add_column(kaggleID, zero_pred)
#pred.rename({'X2':'Prediction'})
#pred_path = r'/Users/yusong/Code/STAT640/Competition/data/output/zero.csv'
#pred.export_csv(pred_path)
#

# ouput for test
length = kaggleID.shape[0]
zero_pred = gl.SArray([6.0809 for i in range(length)])
pred = gl.SFrame.add_column(kaggleID, zero_pred)
pred.rename({'X2':'Prediction'})
pred_path = r'/Users/yusong/Code/STAT640/Competition/data/output/variance.csv'
pred.export_csv(pred_path)