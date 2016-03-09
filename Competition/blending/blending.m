%% Blending
% built the dataset

% X (train) : #ratings * #models
% #ratings of X = 80%
% probe_train : 10% of n * #models
% probe_validate : 10% of n * #models (Never tough except report error)

train = dlmread('../data/trainRatings.csv', ',', 1, 0);
probe_train = dlmread('../data/probe_train.csv', ',', 1, 0);
probe_validate = dlmread('../data/probe_validate.csv', ',', 1, 0);
all = dlmread('../data/ratings.csv', ',', 1, 0);
idmap = dlmread('../data/IDMap.csv',',', 1, 0);

files = dir('./models/*.mat');
n_train = size(train,1);
n_models = size(files,1);
n_validate = size(probe_validate,1);
n_kaggle = size(idmap,1);
X = zeros(n_train, n_models);
y = train(:, 3);

rmse_list = zeros(n_models,1);
probe_mat = zeros(n_validate, n_models);
test_mat = zeros(n_validate, n_models);
kaggle_mat = zeros(n_kaggle, n_models);

for i = 1:n_models
    file = files(i);
    path = strcat('./models/',file.name);
    model = load(path);
    % fill out X
    X(:,i) = model.pred_train;
    probe_mat(:,i) = model.pred_probe;
    test_mat(:,i) = model.pred_test;
    rmse_list(i) = model.rmse;
    kaggle_mat(:,i) = model.pred_kaggle;
%     pred = [(1:n_kaggle)', model.pred_kaggle];
%     headers = {'ID', 'Prediction'};
%     csv_name = strcat('./models/',file.name(1:end-3),'csv');
%     csvwrite_with_headers(csv_name,pred,headers);
%     
 
end
%%
% inverse the wrong mat file
% files = dir('./models/glmodels/*.mat');    
% n_models = size(files,1);
% for i = 1:n_models
%     file = files(i);
%     path = strcat('./models/glmodels/',file.name);
%     model = load(path); 
%     pred_train = model.pred_train';
%     pred_probe = model.pred_probe';
%     pred_kaggle = model.pred_kaggle';
%     pred_test = model.pred_test';
%     rmse = model.rmse;
%     save(strcat('./models/',file.name),'pred_train', 'pred_probe', 'pred_test', 'pred_kaggle', 'rmse' );
% 
% end


%% validation set blending
% substract the mean
% rmean = mean(train(:,3))
% rvar = var(train(:,3))
% y = probe_train(:,3);
% X_centered = probe_mat - rmean;
% y_centered = probe_train(:,3) - rmean;
% lambda = 0;
% coef = zeros(n_validate, n_models);
% pred = zeros(n_validate, 1);
% for i = 1:n_validate
%     yy = y_centered(i);
%     xx = X_centered(i,:);
%     ww = ridge(yy, xx, lambda);
%     coef(i,:) = ww';
%     pred(i,1) = xx*ww;
% end
% 

%% validation set simple averaging

% pred_mean = mean(probe_mat,2);
% rmse(probe_train(:,3), pred_mean)

%% Ridge regression
% lambda = 0:1:10;
xmean = mean(probe_mat(:));
xvar = var(probe_mat(:));
% prediction variance is too small ?
% normalize ?
yc = (probe_train(:,3) - xmean)/(xvar)^0.5;
xc = (probe_mat - xmean)/(xvar)^0.5;
% b = ridge(yc, xc, lambda);
% try elstic net
b = lassoglm(xc, yc);

pred = xc*b*(xvar)^0.5 + xmean ;
for i =1:size(b,2)
    rrmse(i) = rmse(probe_train(:,3), pred(:,i));
end
figure;
plot(rrmse);
% 
% figure;
% plot(lambda,b,'LineWidth',2);
% grid on
% xlabel('Ridge Parameter');
% ylabel('Standardized Coefficient');
% title('{\bf Ridge Trace}');

xc = (kaggle_mat - xmean)/(xvar)^0.5;
pkaggle = xc*b(:,1)*(xvar)^0.5 + xmean ;

%% Testset Blending
% true y unknow
quizvar = var(y);
xy = zeros(n_models, 1);
for i = 1:n_models
    xy(i) = 0.5 * (quizvar + var(xc(:,i))   - sum(quizrmse));
end
bb = (xc'*xc)^(-1)*xy;



%% Write to csv

% y_blend = probe_mat * b;
% y_test = test_mat * b;
% rmse(y_blend, probe_train(:,3))
% rmse(y_test, probe_validate(:,3))

pkaggle(pkaggle<1.5)=1;
pkaggle(pkaggle>9.5)=10;

pred = [(1:n_kaggle)', pkaggle];
headers = {'ID', 'Prediction'};
csvwrite_with_headers('../data/lineaer_combine_9models.csv',pred,headers);
