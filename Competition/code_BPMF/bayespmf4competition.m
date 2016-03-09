%% Bayesan PMF for STAT640 competition
%

rand('state',0);
randn('state',0);

% Triplets: {user_id, profile_id, rating} 
train_vec = dlmread('../data/trainRatings.csv', ',', 1, 0); 
probe_vec = dlmread('../data/testRatings.csv', ',', 1, 0); 
% {user_id, profile_id, kaggle_id} 
idmap = dlmread('../data/IDMap.csv',',', 1, 0);

epoch=1; 
% maxepoch=100; 
maxepoch=20; 


iter=0; 
num_m = 10000;
num_p = 10000;
num_feat = 30;

% Initialize hierarchical priors 
beta=2; % observation noise (precision) default 2
mu_u = zeros(num_feat,1);
mu_m = zeros(num_feat,1);
alpha_u = eye(num_feat);
alpha_m = eye(num_feat);  

% parameters of Inv-Whishart distribution (see paper for details) 
WI_u = eye(num_feat);
b0_u = 2;
df_u = num_feat;
mu0_u = zeros(num_feat,1);

WI_m = eye(num_feat);
b0_m = 2;
df_m = num_feat;
mu0_m = zeros(num_feat,1);

mean_rating = mean(train_vec(:,3));
ratings_test = double(probe_vec(:,3));
ratings_train = double(train_vec(:,3));

pairs_tr = length(train_vec);
pairs_pr = length(probe_vec);

% Initializing Bayesian PMF using MAP solution found by PMF
fprintf(1,'Initializing Bayesian PMF using MAP solution found by PMF \n'); 
makeRatingMatrix

load pmf4competition_weight_final
err_test = cell(maxepoch,1);

w1_P1_sample = w1_P1; 
w1_M1_sample = w1_M1; 
clear w1_P1 w1_M1;

% Initialization using MAP solution found by PMF. 
% Do simple fit
mu_u = mean(w1_P1_sample)';
d=num_feat;
alpha_u = inv(cov(w1_P1_sample));

mu_m = mean(w1_M1_sample)';
alpha_m = inv(cov(w1_P1_sample));

count=count';
probe_rat_all = pred(w1_M1_sample,w1_P1_sample,probe_vec,mean_rating);
counter_prob=1; 
test_rat_all = pred(w1_M1_sample,w1_P1_sample,idmap,mean_rating);
train_rat_all = pred(w1_M1_sample,w1_P1_sample,train_vec,mean_rating);


%%

for epoch = epoch:maxepoch

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% Sample from movie hyperparams (see paper for details)  
  N = size(w1_M1_sample,1);
  x_bar = mean(w1_M1_sample)'; 
  S_bar = cov(w1_M1_sample); 

  WI_post = inv(inv(WI_m) + N/1*S_bar + ...
            N*b0_m*(mu0_m - x_bar)*(mu0_m - x_bar)'/(1*(b0_m+N)));
  WI_post = (WI_post + WI_post')/2;

  df_mpost = df_m+N;
  alpha_m = wishrnd(WI_post,df_mpost);   
  mu_temp = (b0_m*mu0_m + N*x_bar)/(b0_m+N);  
  lam = chol( inv((b0_m+N)*alpha_m) ); lam=lam';
  mu_m = lam*randn(num_feat,1)+mu_temp;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% Sample from user hyperparams
  N = size(w1_P1_sample,1);
  x_bar = mean(w1_P1_sample)';
  S_bar = cov(w1_P1_sample);

  WI_post = inv(inv(WI_u) + N/1*S_bar + ...
            N*b0_u*(mu0_u - x_bar)*(mu0_u - x_bar)'/(1*(b0_u+N)));
  WI_post = (WI_post + WI_post')/2;
  df_mpost = df_u+N;
  alpha_u = wishrnd(WI_post,df_mpost);
  mu_temp = (b0_u*mu0_u + N*x_bar)/(b0_u+N);
  lam = chol( inv((b0_u+N)*alpha_u) ); lam=lam';
  mu_u = lam*randn(num_feat,1)+mu_temp;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Start doing Gibbs updates over user and 
  % movie feature vectors given hyperparams.  

  for gibbs=1:2 
    fprintf(1,'\t\t Gibbs sampling %d \r', gibbs);

    %%% Infer posterior distribution over all movie feature vectors 
    count=count';
    for mm=1:num_m
%        fprintf(1,'movie =%d\r',mm);
       ff = find(count(:,mm)>0);
       MM = w1_P1_sample(ff,:);
       rr = count(ff,mm)-mean_rating;
       covar = inv((alpha_m+beta*MM'*MM));
       mean_m = covar * (beta*MM'*rr+alpha_m*mu_m);
       lam = chol(covar); lam=lam'; 
       w1_M1_sample(mm,:) = lam*randn(num_feat,1)+mean_m;
     end

    %%% Infer posterior distribution over all user feature vectors 
     count=count';
     for uu=1:num_p
%        fprintf(1,'user  =%d\r',uu);
       ff = find(count(:,uu)>0);
       MM = w1_M1_sample(ff,:);
       rr = count(ff,uu)-mean_rating;
       covar = inv((alpha_u+beta*MM'*MM));
       mean_u = covar * (beta*MM'*rr+alpha_u*mu_u);
       lam = chol(covar); lam=lam'; 
       w1_P1_sample(uu,:) = lam*randn(num_feat,1)+mean_u;
     end
   end
   
   train_rat = pred(w1_M1_sample,w1_P1_sample,train_vec,mean_rating);
   train_rat_all = (counter_prob*train_rat_all + train_rat)/(counter_prob+1);
       
   probe_rat = pred(w1_M1_sample,w1_P1_sample,probe_vec,mean_rating);
   probe_rat_all = (counter_prob*probe_rat_all + probe_rat)/(counter_prob+1);
   
   %%% predict the kaggle test set
   test_rat = pred(w1_M1_sample,w1_P1_sample,idmap,mean_rating);
   test_rat_all = (counter_prob*test_rat_all + test_rat)/(counter_prob+1);
   
   counter_prob=counter_prob+1;

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%% Make predictions on the training and validation data %%%%%%%
   temp = (ratings_test - probe_rat_all).^2;
   err = sqrt( sum(temp)/pairs_pr);

   temp2 = (ratings_train - train_rat_all).^2;
   err2 = sqrt( sum(temp2)/pairs_tr);
   
   iter=iter+1;
   overall_err(iter,1)=err;
   overall_err(iter,2)=err2;

  fprintf(1, '\nEpoch %d \t Average Training RMSE %6.4f \n', epoch, err2);
  fprintf(1, '\nEpoch %d \t Average Test RMSE %6.4f \n', epoch, err);

end 

% save bpmf_kaggle_ratmat10 test_rat_all

