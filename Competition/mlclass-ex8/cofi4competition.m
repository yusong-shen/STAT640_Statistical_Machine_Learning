% CF for STAT640 Competition

clear all;

ratings = dlmread('../data/trainRatings.csv', ',', 1, 0);
probe_train = dlmread('../data/probe_train.csv', ',', 1, 0);
probe_validate = dlmread('../data/probe_validate.csv', ',', 1, 0);
idmap = dlmread('../data/IDMap.csv',',', 1, 0);

Y = sparse(ratings(:,2),ratings(:,1),ratings(:,3));
R = (Y~=0);
%% training
%  Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);


%  Useful Values
num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = 10;

% Set Initial Parameters (Theta, X)
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);

initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 150);

% Set Regularization
lambda = 20;
theta = fmincg (@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies, ...
                                num_features, lambda)), ...
                initial_parameters, options);

% Unfold the returned theta back into U and W
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), ...
                num_users, num_features);

fprintf('Recommender system learning completed.\n');


%% prediction

p = X * Theta';
% predmat = p;
predmat = p + repmat(Ymean,1,num_users);

% trRMSE = rmse(Y, predmat);

testIDmap = dlmread('../data/testIDmap.csv',',', 1, 0);
testN = size(testIDmap,1);



% 
% headers = {'ID', 'Prediction'};
% csvwrite_with_headers('../data/cofi.csv',pred,headers);

%%
% pred_train : n_train * 1
pred_train = zeros(size(ratings,1),1);
for i = 1:size(ratings,1)
    pred_train(i) = predmat(ratings(i,2), ratings(i,1));
end

pred_probe = zeros(size(probe_train,1),1);

for i = 1:size(probe_train,1)
    pred_probe(i) = predmat(probe_train(i,2), probe_train(i,1));
end

pred_test = zeros(size(probe_validate,1),1);

for i = 1:size(probe_validate,1)
    pred_test(i) = predmat(probe_validate(i,2), probe_validate(i,1));
end


pred_kaggle = zeros(size(idmap,1),1);
for i = 1:size(idmap,1)
    pred_kaggle(i) = predmat(idmap(i,2), idmap(i,1));
end

rmse = rmse(probe_train(:,3), pred_probe)
filename = sprintf('/Users/yusong/Code/STAT640/Competition/blending/models/cofi_D%d',D);
save(filename, 'pred_train', 'pred_probe', 'pred_test', 'pred_kaggle', 'rmse');

