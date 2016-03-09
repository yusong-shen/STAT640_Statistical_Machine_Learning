# -*- coding: utf-8 -*-
"""

STAT 640
Exploratary analysis for Competition

Created on Tue Nov 17 14:03:14 2015

@author: yusong
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.io as sio
import time

# Measure the running time
start = time.time()

rpath = r'/Users/yusong/Code/STAT640/Competition/data/ratings.csv'
rmat_path = r'/Users/yusong/Code/STAT640/Competition/data/rmat.csv'
idmap_path = r'/Users/yusong/Code/STAT640/Competition/data/IDMap.csv'
kaggleid_path = r'/Users/yusong/Code/STAT640/Competition/data/KaggleID.csv'
gender_path = r'/Users/yusong/Code/STAT640/Competition/data/gender.csv'

#ratings = pd.read_csv(rpath)
rmat = pd.read_csv(rmat_path)


# Extract the ratings from the DataFrame
all_ratings = np.ndarray.flatten(rmat.values)
all_ratings = all_ratings[all_ratings!=0]
ratings = pd.Series(all_ratings)

# Plot histogram and density.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
ratings.plot(kind='density', ax=ax1, grid=False)
ax1.set_ylim(0,0.5)
ax1.set_xlim(-1, 11)

# Plot histogram
ratings.plot(kind='hist', ax=ax2, bins=10, grid=False)
ax2.set_xlim(-1, 11)
plt.show()
plt.savefig('density_hist_of_ratings.png')


x_path = r'/Users/yusong/Code/STAT640/Competition/data/X.mat'
theta_path = r'/Users/yusong/Code/STAT640/Competition/data/Theta.mat'
Xmat = sio.loadmat(x_path)
Theta = sio.loadmat(theta_path)
gender = pd.read_csv(gender_path)

end = time.time()
print "time elapse : %6.2f secondes"%(end-start)