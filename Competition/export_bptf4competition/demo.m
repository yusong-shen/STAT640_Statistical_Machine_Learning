% (C) Copyright 2011, Liang Xiong (lxiong[at]cs[dot]cmu[dot]edu)
% 
% This piece of software is free for research purposes. 
% We hope it is helpful but do not privide any warranty.
% If you encountered any problems please contact the author.
%%
build;

cd ./lib
build;
cd ..
addpath ./lib

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

D = 200;

pn = 50e-3;
max_iter = 100;
learn_rate = 1e-3;
n_sample = 150;

[U, V, dummy, r_pmf ,yTe] = PMF_Grad(CTr, CTe, D, ...
    struct('ridge',pn,'learn_rate',learn_rate,'range',[1,10],'max_iter',max_iter));
fprintf('PMF: %.4f\n', r_pmf);

yTr = PMF_Reconstruct(CTr.subs, U, V);
rmseTr = RMSE(yTr - TTr.vals);
pred_train = yTr;

n_probe = size(yTe,1);
pred_probe = yTe(1:n_probe/2, :);
pred_test = yTe(n_probe/2+1:end, :);

pred_kaggle = PMF_Reconstruct(CTs.subs, U, V);
rmse = r_pmf;
filename = sprintf('/Users/yusong/Code/STAT640/Competition/blending/models/pmf_D%d',D);
save(filename, 'pred_train', 'pred_probe', 'pred_test', 'pred_kaggle', 'rmse');


% save pmf200pred yTe
% 
% ff = yTe>9.5; yTe(ff)=10; % Clip predictions 
% ff = find(yTe<1.5); yTe(ff)=1;
% % 
% testN = size(idmap_value,1);
% pred = [(1:testN)', yTe];
% headers = {'ID', 'Prediction'};
% csvwrite_with_headers('../data/pmf_f200_chop.csv',pred,headers);


%%
% alpha = 2;
% [Us_bpmf, Vs_bpmf] = BPMF(CTr, CTe, D, alpha, [], {U,V}, ...
%     struct('max_iter',n_sample,'n_sample',n_sample,'save_sample',false, 'run_name','alpha2'));
% [Y_bpmf] = BPMF_Predict(Us_bpmf, Vs_bpmf, D, CTe, [1 10]);
% r_bpmf = RMSE(Y_bpmf.vals - CTe.vals);
% fprintf('BPMF: %.4f\n', r_bpmf);
% save bpmf_f20 Us_bpmf Vs_bpmf Y_bpmf r_bpmf


%%
% 
% [Us_bptf Vs_bptf Ts_bptf] = BPTF(TTr, TTe, D, struct('Walpha',alpha, 'nuAlpha',1), ...
%     {U,V,ones(D,TTr.size(3))}, struct('max_iter',n_sample,'n_sample',n_sample,'save_sample',false,'run_name','alpha2-1'));
% [Y_bptf] = BPTF_Predict(Us_bptf,Vs_bptf,Ts_bptf,D,TTe,[1 10]);
% r_bptf = RMSE(Y_bptf.vals-TTe.vals);
% fprintf('BPTF: %.4f\n', r_bptf);
% % save bptf_f200 -v7.3 Us_bptf Vs_bptf Ts_bptf Y_bptf
% 
% % r = [r_pmf r_bpmf r_bptf]
% r = [r_pmf r_bptf]

%% predict in kaggle id
% D = 100;
% [Y_bptf] = BPTF_Predict(Us_bptf,Vs_bptf,Ts_bptf,D,idmap,[1 10]);
% 
% testN = size(idmap_value,1);
% yTe = Y_bptf.vals;
% 
% ff = yTe>9.5; yTe(ff)=10; % Clip predictions 
% ff = find(yTe<1.5); yTe(ff)=1;
% % 
% pred = [(1:testN)', yTe];
% headers = {'ID', 'Prediction'};
% csvwrite_with_headers('../data/bptf_f200_chop2.csv',pred,headers);

