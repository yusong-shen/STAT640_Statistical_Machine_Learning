%% Split the dataset 


% train = dlmread('../data/trainRatings.csv', ',', 1, 0);
% validate = dlmread('../data/testRatings.csv', ',', 1, 0);
% all = dlmread('../data/ratings.csv', ',', 1, 0);
% idmap = dlmread('../data/IDMap.csv',',', 1, 0);

% n_probe = size(validate,1);
% probe_train = validate(1: n_probe/2, :);
% probe_validate = validate(n_probe/2+1:end, :);


%%
addpath ../export_bptf4competition
load TTr
load TTe
load idmap
TTr = spTensor(TTr.subs, TTr.vals, TTr.size);
TTe = spTensor(TTe.subs, TTe.vals, TTe.size);
CTr = TTr.Reduce(1:2);
CTe = TTe.Reduce(1:2);
idmap_value = ones(size(idmap_kaggle,1),1);
idmap = spTensor(idmap_kaggle, idmap_value, [10000,10000,3]);
TTs = idmap;
CTs = idmap.Reduce(1:2);

n_probe = size(TTe.vals, 1);

%% write to csv

% headers = {'UserID', 'ProfileID', 'Rating'};
% csvwrite_with_headers('../data/probe_train.csv',probe_train,headers);
% csvwrite_with_headers('../data/probe_validate.csv',probe_validate,headers);

%% bptf D30
load bptf_f20
D = 30;
[pred_train,U,V,T, rmse] = BPTF_Predict(Us_bptf,Vs_bptf,Ts_bptf,D,TTr,[1 10]);
pred_train = pred_train.vals;

pred = Y_bptf.vals;
pred_probe = pred(1:n_probe/2, :);
pred_test = pred(n_probe/2+1:end, :);

rmse = RMSE(pred_probe - probe_train(:,3))

pred_kaggle = BPTF_Predict(Us_bptf,Vs_bptf,Ts_bptf,D,idmap,[1 10]);
pred_kaggle = pred_kaggle.vals;

filename = sprintf('/Users/yusong/Code/STAT640/Competition/blending/models/bptf_D%d',D);
save(filename, 'pred_train', 'pred_probe', 'pred_test', 'pred_kaggle', 'rmse');

%% bptf D100
load bptf_f100
D = 100;
[pred_train] = BPTF_Predict(Us_bptf,Vs_bptf,Ts_bptf,D,TTr,[1 10]);
pred_train = pred_train.vals;

pred = Y_bptf.vals;
pred_probe = pred(1:n_probe/2, :);
pred_test = pred(n_probe/2+1:end, :);

rmse = RMSE(pred_probe - probe_train(:,3))

pred_kaggle = BPTF_Predict(Us_bptf,Vs_bptf,Ts_bptf,D,idmap,[1 10]);
pred_kaggle = pred_kaggle.vals;

filename = sprintf('/Users/yusong/Code/STAT640/Competition/blending/models/bptf_D%d',D);
save(filename, 'pred_train', 'pred_probe', 'pred_test', 'pred_kaggle', 'rmse');


%% bpft D200
load bptf_f200
D = 200;
[pred_train] = BPTF_Predict(Us_bptf,Vs_bptf,Ts_bptf,D,TTr,[1 10]);
pred_train = pred_train.vals;

pred = Y_bptf.vals;
pred_probe = pred(1:n_probe/2, :);
pred_test = pred(n_probe/2+1:end, :);

rmse = RMSE(pred_probe - probe_train(:,3))

pred_kaggle = BPTF_Predict(Us_bptf,Vs_bptf,Ts_bptf,D,idmap,[1 10]);
pred_kaggle = pred_kaggle.vals;

filename = sprintf('/Users/yusong/Code/STAT640/Competition/blending/models/bptf_D%d',D);
save(filename, 'pred_train', 'pred_probe', 'pred_test', 'pred_kaggle', 'rmse');



%%
% testN = size(idmap,1);
% pred = [(1:testN)', pred_kaggle];
% headers = {'ID', 'Prediction'};
% csvwrite_with_headers('../data/pmf_f30_all_data.csv',pred,headers);
