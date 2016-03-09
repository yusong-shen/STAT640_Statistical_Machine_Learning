# -*- coding: utf-8 -*-
"""

Helper functions
Created on Mon Sep 28 16:52:31 2015

@author: yusong
"""
import graphlab as gl
import graphlab.numpy


    
def modify_rating(ratings, lower=1.5, higher=9.5):
    """
    make sure the rating range from 1 - 10,
    assume ratings is a SArray
    """
    rs = gl.numpy.array(ratings)
    rs[rs<lower] = 1
    rs[rs>higher] = 10
    return gl.SArray(rs)
 
def discrete_ratings(ratings, threshold=0.5, lower=1.5, higher=9.5):
    """
    change rating from 3.8 - 4.2 to 4
    assume ratings is a SArray
    """
    rs = gl.numpy.array(ratings)    
    for i in range(2,10):
        rs[(rs>(i-threshold)) & (rs<=(i+threshold))] = i
    rs[rs<=lower] = 1
    rs[rs>higher] = 10        
    return gl.SArray(rs) 