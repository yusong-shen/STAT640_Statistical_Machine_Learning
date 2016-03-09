% matrix factorization for STAT640 competition

clear all
% set parameter
steps = 20;
alpha = 0.0002;
beta = 0.02;

% mf
ratings = dlmread('ratings.csv', ',', 1, 0);
testIDmap = dlmread('testIDmap.csv', ',', 1, 0);
R = sparse(ratings(:,1), ratings(:,2), ratings(:,3));

N = size(R,1);
M = size(R,2);
K = 20;

P = randi(10,N,K);
Q = randi(10,M,K);
% 
% [Pfinal, Qfinal ] = ...
% matrixFactorization( R, P, Q, K, steps, alpha, beta );

% Toolbox
% non-negative matrix factorization 
opts = statset('Display','iter', 'MaxIter',100,'TolFun',1.00e-08,'TolX',1.00e-08);
[Pfinal, Qfinal] = nnmf(R, K, 'options', opts);

% % 
% [Pfinal, Qfinal, numIter, tElapsed, finalResidual] = ...
%     sparsenmfnnls(R, K);
Rpred = Pfinal*Qfinal;

testN = size(testIDmap,1);
% pred 
pred = [(1:testN)',zeros(testN,1)];
for i = 1:testN
    pred(i,2) = Rpred(testIDmap(i,1),testIDmap(i,2));

end

groundTruth = dlmread('groundTruth.csv', ',', 1, 0);
mfRMSE = rmse(groundTruth(:,2), pred(:,2));
