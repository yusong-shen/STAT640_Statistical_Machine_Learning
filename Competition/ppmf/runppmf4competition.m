%%
clear;

% Triplets: {user_id, profile_id, rating} 
train_vec = dlmread('../data/trainRatings.csv', ',', 1, 0); 
% probe_vec = dlmread('../data/testRatings.csv', ',', 1, 0); 
% {user_id, profile_id, kaggle_id}
idmap = dlmread('../data/IDMap.csv', ',', 1, 0);
[n_idmap, m_idmap] = size(idmap);
mask_kaggle = sparse(idmap(:,1), idmap(:,2), ones(n_idmap));

R_train = sparse(train_vec(:,1),train_vec(:,2),train_vec(:,3));
% R_test = sparse(probe_vec(:,1),probe_vec(:,2),probe_vec(:,3));
% R_v = sparse(probe_vec(:,1),probe_vec(:,2),probe_vec(:,3));
mask_train = (R_train~=0);
% mask_test = (R_test~=0);
% mask_v = (R_v~=0);

clear train_vec;
% cleatr probe_vec;
clear idmap;

[N,M]=size(R_train);

k=10;

% set the init type
inittype=2;


% perform ppmf given the initial value for variational parameters
if inittype==2;
    init.Lambda1=rand(k,N);
    init.Nu1=rand(k,N);
    init.Lambda2=rand(k,M);
    init.Nu2=rand(k,M);
    [mu1,Sigma1,mu2,Sigma2,tau,Lambda1,Nu1,Lambda2,Nu2,mv]=ppmfLearn(R_train,mask_train,R_train,mask_train,inittype,init);
    % prediction on the whole matrix
%     [R_pred,rmse]=ppmfPred(Lambda1,Lambda2,mv,R_test,mask_test);
    [R_kaggle,dummy] = ppmfPred(Lambda1,Lambda2,mv,mask_kaggle,mask_kaggle);
% perform ppmf given the initial value for model parameters
elseif inittype==1
    init.mu1=rand(k,1);
    init.Sigma1=rand(k,k);
    init.mu2=rand(k,1);
    init.Sigma2=rand(k,k);
    init.tau=1;
    [mu1,Sigma1,mu2,Sigma2,tau,Lambda1,Nu1,Lambda2,Nu2,mv]=ppmfLearn(R_train,mask_train,R_v,mask_v,inittype,init);
    % prediction on the whole matrix
    [R_pred,rmse]=ppmfPred(Lambda1,Lambda2,mv,R_test,mask_test);
%     [R_kaggle,dummy] = ppmfPred(Lambda1,Lambda2,mv,mask_kaggle,mask_kaggle);

end


