%%


ratings = dlmread('../data/trainRatings.csv', ',', 1, 0);
probe_train = dlmread('../data/probe_train.csv', ',', 1, 0);
probe_validate = dlmread('../data/probe_validate.csv', ',', 1, 0);

n_kaggle = 500000;
n_probe_train = size(probe_train,1);
n_models = 4;

cofi = load('./models/cofi_D10.mat');
pmf200 = load('./models/pmf_D200.mat');
bptf200 = load('./models/bptf_D200.mat');
bptf100 = load('./models/bptf_D100.mat');
bptf30 = load('./models/bptf_D30.mat');




final_pred_probe = 1.0/n_models*(pmf200.pred_probe...
    + bptf200.pred_probe + bptf100.pred_probe + bptf30.pred_probe);

final_pred_kaggle = 1.0/n_models*(pmf200.pred_kaggle...
    + bptf200.pred_kaggle + bptf100.pred_kaggle + bptf30.pred_kaggle);

rmse(final_pred_probe, probe_train(:,3))
rmse(bptf30.pred_probe, probe_train(:,3))
rmse(cofi.pred_train, ratings(:,3))


% %%
% ff = final_pred_kaggle>9.5; final_pred_kaggle(ff)=10; % Clip predictions 
% ff = find(final_pred_kaggle<1.5); final_pred_kaggle(ff)=1;
%  
% testN = n_kaggle;
% pred = [(1:testN)', final_pred_kaggle];
% headers = {'ID', 'Prediction'};
% csvwrite_with_headers('../data/mf_combine.csv',pred,headers);
% 
