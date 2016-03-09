
%% PMF for STAT640 competition
%

rand('state',0); 
randn('state',0); 

% Triplets: {user_id, profile_id, rating} 
train_vec = dlmread('../data/ratings.csv', ',', 1, 0); 
probe_vec = dlmread('../data/testRatings.csv', ',', 1, 0); 
idmap = dlmread('../data/IDMap.csv',',', 1, 0);
groundTruth = dlmread('../data/groundTruth.csv', ',', 1, 0);

epsilon=50; % Learning rate 
lambda  = 0.2; % Regularization parameter default=0.01
momentum=0.9;  % default=0.8

epoch=1; 
% maxepoch=100; 
maxepoch=30; 


mean_rating = mean(train_vec(:,3)); 

pairs_tr = length(train_vec); % training data 
pairs_pr = length(probe_vec); % validation data 

numbatches= 9; % Number of batches  
num_m = 10000;  % Number of profiles 
num_p = 10000;  % Number of users 
num_feat = 30; % Rank 10 decomposition 

w1_M1     = 0.1*randn(num_m, num_feat); % Movie feature vectors
w1_P1     = 0.1*randn(num_p, num_feat); % User feature vecators
w1_M1_inc = zeros(num_m, num_feat);
w1_P1_inc = zeros(num_p, num_feat);

%%
for epoch = epoch:maxepoch
  % shuffle the training set
  rr = randperm(pairs_tr);
  train_vec = train_vec(rr,:);
  clear rr 

  for batch = 1:numbatches
    fprintf(1,'epoch %d batch %d \r',epoch,batch);
    N=100000; % number training triplets per batch 

    aa_p   = double(train_vec((batch-1)*N+1:batch*N,1));
    aa_m   = double(train_vec((batch-1)*N+1:batch*N,2));
    rating = double(train_vec((batch-1)*N+1:batch*N,3));

    rating = rating-mean_rating; % Default prediction is the mean rating. 

    %%%%%%%%%%%%%% Compute Predictions %%%%%%%%%%%%%%%%%
    pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2);
    f = sum( (pred_out - rating).^2 + ...
        0.5*lambda*( sum( (w1_M1(aa_m,:).^2 + w1_P1(aa_p,:).^2),2)));

    %%%%%%%%%%%%%% Compute Gradients %%%%%%%%%%%%%%%%%%%
    IO = repmat(2*(pred_out - rating),1,num_feat);
    Ix_m=IO.*w1_P1(aa_p,:) + lambda*w1_M1(aa_m,:);
    Ix_p=IO.*w1_M1(aa_m,:) + lambda*w1_P1(aa_p,:);

    dw1_M1 = zeros(num_m,num_feat);
    dw1_P1 = zeros(num_p,num_feat);

    for ii=1:N
      dw1_M1(aa_m(ii),:) =  dw1_M1(aa_m(ii),:) +  Ix_m(ii,:);
      dw1_P1(aa_p(ii),:) =  dw1_P1(aa_p(ii),:) +  Ix_p(ii,:);
    end

    %%%% Update movie and user features %%%%%%%%%%%

    w1_M1_inc = momentum*w1_M1_inc + epsilon*dw1_M1/N;
    w1_M1 =  w1_M1 - w1_M1_inc;

    w1_P1_inc = momentum*w1_P1_inc + epsilon*dw1_P1/N;
    w1_P1 =  w1_P1 - w1_P1_inc;
  end 

  %%%%%%%%%%%%%% Compute Predictions after Paramete Updates %%%%%%%%%%%%%%%%%
  pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2);
  f_s = sum( (pred_out - rating).^2 + ...
        0.5*lambda*( sum( (w1_M1(aa_m,:).^2 + w1_P1(aa_p,:).^2),2)));
  err_train(epoch) = sqrt(f_s/N);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% Compute predictions on the validation set %%%%%%%%%%%%%%%%%%%%%% 
  NN=pairs_pr;

  aa_p = double(probe_vec(:,1));
  aa_m = double(probe_vec(:,2));
  rating = double(probe_vec(:,3));

  pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2) + mean_rating;
  ff = pred_out>9.5; pred_out(ff)=10; % Clip predictions 
  ff = find(pred_out<1.5); pred_out(ff)=1;

  err_valid(epoch) = sqrt(sum((pred_out- rating).^2)/NN);
  fprintf(1, 'epoch %4i batch %4i Training RMSE %6.4f  Test RMSE %6.4f  \n', ...
              epoch, batch, err_train(epoch), err_valid(epoch));
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%% Compute predictions on the kaggle test set %%%%%%%%%%%%%%%%%%%%%% 

  aa_p = double(idmap(:,1));
  aa_m = double(idmap(:,2));

  pred_kaggle = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2) + mean_rating;
  ff = pred_kaggle>9.5; pred_kaggle(ff)=10; % Clip predictions 
  ff = find(pred_kaggle<1.5); pred_kaggle(ff)=1;

   
   
  if (rem(epoch,10))==0
     save pmf4competition_weight_final w1_M1 w1_P1
     
  end

end 


save pmf4kagglepred_final2 pred_kaggle
% 
% testN = size(idmap,1);
% pred = [(1:testN)', pred_kaggle];
% headers = {'ID', 'Prediction'};
% csvwrite_with_headers('../data/pmf_f30_all_data.csv',pred,headers);
% 
