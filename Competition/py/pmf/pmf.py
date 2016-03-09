# -*- coding: utf-8 -*-
"""
Probablistic Matrix Factorization
Reference : https://pymc-devs.github.io/pymc3/pmf-pymc/

Created on Fri Nov 27 20:47:52 2015

@author: yusong
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import logging
import pymc3 as pm
import theano
import scipy as sp
import os

#data_path = r'/Users/yusong/Code/STAT640/Competition/data/jester-dataset-v1-dense-first-1000.csv'
data_path = r'/Users/yusong/Code/STAT640/Competition/data/rmat.csv'

data = pd.read_csv(data_path)
#data.head()
#
## Extract the ratings from the DataFrame
#all_ratings = np.ndarray.flatten(data.values)
#ratings = pd.Series(all_ratings)
#
## Plot histogram and density.
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
#ratings.plot(kind='density', ax=ax1, grid=False)
#ax1.set_ylim(0, 0.08)
#ax1.set_xlim(-11, 11)
#
## Plot histogram
#ratings.plot(kind='hist', ax=ax2, bins=20, grid=False)
#ax2.set_xlim(-11, 11)
#plt.show()
#
#ratings.describe()
#
#joke_means = data.mean(axis=0)
#joke_means.plot(kind='bar', grid=False, figsize=(16, 6),
#                title="Mean Ratings for All 100 Jokes")
#                
#user_means = data.mean(axis=1)
#fig, ax = plt.subplots(figsize=(16, 6))
#user_means.plot(kind='bar', grid=False, ax=ax,
#                title="Mean Ratings for All 1000 Users")
#ax.set_xticklabels('')  # 1000 labels is nonsensical
#fig.show()                
###############################################################################

# Enable on-the-fly graph computations, but ignore 
# absence of intermediate test values.
theano.config.compute_test_value = 'ignore'

# Set up logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class PMF(object):
    """Probabilistic Matrix Factorization model using pymc3."""

    def __init__(self, train, dim, alpha=2, std=0.01, bounds=(-10, 10)):
        """Build the Probabilistic Matrix Factorization model using pymc3.

        :param np.ndarray train: The training data to use for learning the model.
        :param int dim: Dimensionality of the model; number of latent factors.
        :param int alpha: Fixed precision for the likelihood function.
        :param float std: Amount of noise to use for model initialization.
        :param (tuple of int) bounds: (lower, upper) bound of ratings.
            These bounds will simply be used to cap the estimates produced for R.

        """
        self.dim = dim
        self.alpha = alpha
        self.std = np.sqrt(1.0 / alpha)
        self.bounds = bounds
        self.data = train.copy()
        n, m = self.data.shape

        # Perform mean value imputation
        nan_mask = np.isnan(self.data)
        self.data[nan_mask] = self.data[~nan_mask].mean()

        # Low precision reflects uncertainty; prevents overfitting.
        # Set to the mean variance across users and items.
        self.alpha_u = 1 / self.data.var(axis=1).mean()
        self.alpha_v = 1 / self.data.var(axis=0).mean()

        # Specify the model.
        logging.info('building the PMF model')
        with pm.Model() as pmf:
            U = pm.MvNormal(
                'U', mu=0, tau=self.alpha_u * np.eye(dim),
                shape=(n, dim), testval=np.random.randn(n, dim) * std)
            V = pm.MvNormal(
                'V', mu=0, tau=self.alpha_v * np.eye(dim),
                shape=(m, dim), testval=np.random.randn(m, dim) * std)
            R = pm.Normal(
                'R', mu=theano.tensor.dot(U, V.T), tau=self.alpha * np.ones((n, m)),
                observed=self.data)

        logging.info('done building the PMF model') 
        self.model = pmf

    def __str__(self):
        return self.name
        
try:
    import ujson as json
except ImportError:
    import json


# First define functions to save our MAP estimate after it is found.
# We adapt these from `pymc3`'s `backends` module, where the original
# code is used to save the traces from MCMC samples.
def save_np_vars(vars, savedir):
    """Save a dictionary of numpy variables to `savedir`. We assume
    the directory does not exist; an OSError will be raised if it does.
    """
    logging.info('writing numpy vars to directory: %s' % savedir)
    os.mkdir(savedir)
    shapes = {}
    for varname in vars:
        data = vars[varname]
        var_file = os.path.join(savedir, varname + '.txt')
        np.savetxt(var_file, data.reshape(-1, data.size))
        shapes[varname] = data.shape

        ## Store shape information for reloading.
        shape_file = os.path.join(savedir, 'shapes.json')
        with open(shape_file, 'w') as sfh:
            json.dump(shapes, sfh)


def load_np_vars(savedir):
    """Load numpy variables saved with `save_np_vars`."""
    shape_file = os.path.join(savedir, 'shapes.json')
    with open(shape_file, 'r') as sfh:
        shapes = json.load(sfh)

    vars = {}
    for varname, shape in shapes.items():
        var_file = os.path.join(savedir, varname + '.txt')
        vars[varname] = np.loadtxt(var_file).reshape(shape)

    return vars


# Now define the MAP estimation infrastructure.
def _map_dir(self):
    basename = 'pmf-map-d%d' % self.dim
    return os.path.join('data', basename)

def _find_map(self):
    """Find mode of posterior using Powell optimization."""
    tstart = time.time()
    with self.model:
        logging.info('finding PMF MAP using Powell optimization...')
        self._map = pm.find_MAP(fmin=sp.optimize.fmin_powell, disp=True)

    elapsed = int(time.time() - tstart)
    logging.info('found PMF MAP in %d seconds' % elapsed)

    # This is going to take a good deal of time to find, so let's save it.
    save_np_vars(self._map, self.map_dir)

def _load_map(self):
    self._map = load_np_vars(self.map_dir)

def _map(self):
    try:
        return self._map
    except:
        if os.path.isdir(self.map_dir):
            self.load_map()
        else:
            self.find_map()
        return self._map


# Update our class with the new MAP infrastructure.
PMF.find_map = _find_map
PMF.load_map = _load_map
PMF.map_dir = property(_map_dir)
PMF.map = property(_map)        

###############################################################################
# Draw MCMC samples.
def _trace_dir(self):
    basename = 'pmf-mcmc-d%d' % self.dim
    return os.path.join('data', basename)

def _draw_samples(self, nsamples=1000, njobs=2):
    # First make sure the trace_dir does not already exist.
    if os.path.isdir(self.trace_dir):
        raise OSError(
            'trace directory %s already exists. Please move or delete.' % self.trace_dir)
    start = self.map  # use our MAP as the starting point
    with self.model:
        logging.info('drawing %d samples using %d jobs' % (nsamples, njobs))
        step = pm.NUTS(scaling=start)
        backend = pm.backends.Text(self.trace_dir)
        logging.info('backing up trace to directory: %s' % self.trace_dir)
        self.trace = pm.sample(nsamples, step, start=start, njobs=njobs, trace=backend)

def _load_trace(self):
    with self.model:
        self.trace = pm.backends.text.load(self.trace_dir)


# Update our class with the sampling infrastructure.
PMF.trace_dir = property(_trace_dir)
PMF.draw_samples = _draw_samples
PMF.load_trace = _load_trace


###############################################################################
def _predict(self, U, V):
    """Estimate R from the given values of U and V."""
    R = np.dot(U, V.T)
    n, m = R.shape
    sample_R = np.array([
        [np.random.normal(R[i,j], self.std) for j in xrange(m)]
        for i in xrange(n)
    ])

    # bound ratings
    low, high = self.bounds
    sample_R[sample_R < low] = low
    sample_R[sample_R > high] = high
    return sample_R


PMF.predict = _predict

###############################################################################

# Define our evaluation function.
def rmse(test_data, predicted):
    """Calculate root mean squared error.
    Ignoring missing values in the test data.
    """
    I = ~np.isnan(test_data)   # indicator for missing values
    N = I.sum()                # number of non-missing values
    sqerror = abs(test_data - predicted) ** 2  # squared error array
    mse = sqerror[I].sum() / N                 # mean squared error
    return np.sqrt(mse)                        # RMSE
    

###############################################################################
# Define a function for splitting train/test data.
def split_train_test(data, percent_test=10):
    """Split the data into train/test sets.
    :param int percent_test: Percentage of data to use for testing. Default 10.
    """
    n, m = data.shape             # # users, # jokes
    N = n * m                     # # cells in matrix
    test_size = N / percent_test  # use 10% of data as test set
    train_size = N - test_size    # and remainder for training

    # Prepare train/test ndarrays.
    train = data.copy().values
    test = np.ones(data.shape) * np.nan

    # Draw random sample of training data to use for testing.
    tosample = np.where(~np.isnan(train))       # ignore nan values in data
    idx_pairs = zip(tosample[0], tosample[1])   # tuples of row/col index pairs
    indices = np.arange(len(idx_pairs))         # indices of index pairs
    sample = np.random.choice(indices, replace=False, size=test_size)

    # Transfer random sample from train set to test set.
    for idx in sample:
        idx_pair = idx_pairs[idx]
        test[idx_pair] = train[idx_pair]  # transfer to test set
        train[idx_pair] = np.nan          # remove from train set

    # Verify everything worked properly
    assert(np.isnan(train).sum() == test_size)
    assert(np.isnan(test).sum() == train_size)

    # Finally, hash the indices and save the train/test sets.
    index_string = ''.join(map(str, np.sort(sample)))
    name = hashlib.sha1(index_string).hexdigest()
    savedir = os.path.join('data', name)
    save_np_vars({'train': train, 'test': test}, savedir)

    # Return train set, test set, and unique hash of indices.
    return train, test, name


def load_train_test(name):
    """Load the train/test sets."""
    savedir = os.path.join('data', name)
    vars = load_np_vars(savedir)
    return vars['train'], vars['test']
    
def eval_map(pmf_model, train, test):
    U = pmf_model.map['U']
    V = pmf_model.map['V']

    # Make predictions and calculate RMSE on train & test sets.
    predictions = pmf_model.predict(U, V)
    train_rmse = rmse(train, predictions)
    test_rmse = rmse(test, predictions)
    overfit = test_rmse - train_rmse

    # Print report.
    print 'PMF MAP training RMSE: %.5f' % train_rmse
    print 'PMF MAP testing RMSE:  %.5f' % test_rmse
    print 'Train/test difference: %.5f' % overfit

    return test_rmse


# Add eval function to PMF class.
PMF.eval_map = eval_map
#train, test = load_train_test('6bb8d06c69c0666e6da14c094d4320d115f1ffc8')
train, test, name = split_train_test(data)

###############################################################################
# We use a fixed precision for the likelihood.
# This reflects uncertainty in the dot product.
# We choose 2 in the footsteps Salakhutdinov
# Mnihof.
ALPHA = 2

# The dimensionality D; the number of latent factors.
# We can adjust this higher to try to capture more subtle
# characteristics of each joke. However, the higher it is,
# the more expensive our inference procedures will be.
# Specifically, we have D(N + M) latent variables. For our
# Jester dataset, this means we have D(1100), so for 5
# dimensions, we are sampling 5500 latent variables.
DIM = 5


pmf = PMF(train, DIM, ALPHA, std=0.05)
###############################################################################
#Predictions Using MAP
# Find MAP for PMF.
pmf.find_map()
# pmf.load_map()

# Evaluate PMF MAP estimates.
pmf_map_rmse = pmf.eval_map(train, test)
