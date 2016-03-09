The code is for parametric probabistic matrix factorization (PPMF) in the paper "H. Shan and A. Banerjee. Generalized Probabilistic Matrix Factorization for Collaborative Filtering. ICDM, 2010".

runppmf.m:	An example on how to run the code.
ppmfLearn.m:	learning process of ppmf. It calls ppmfEstep.m and ppmfMstep.m.
ppmfEstep.m:	Variational E-step.
ppmfMstep.m:	Variational M-step.
ppmfPred.m:	predict on the test set.
data.mat:	Sample data (movielens data).