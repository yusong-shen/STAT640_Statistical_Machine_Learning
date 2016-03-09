%% Reading PowerGraph output using Matlab
% The output of the program are three matrices U,V and T of size dim1 X D, dim2 X D and dim3 X D.
% The output is generated to a file named [inputfile].out
% A=mmread('wals-20-11.out.V')




%% predict in kaggle id
idmap = dlmread('../data/IDMap.csv',',', 1, 0);

R = U * V;

testN = size(idmap,1);
yTe = ones(testN,1);
for i = 1:testN
    u = size(idmap,1);
    p = size(idmap,2);
    yTe(i) = R(u,p);
end


ff = yTe>9.5; yTe(ff)=10; % Clip predictions 
ff = find(yTe<1.5); yTe(ff)=1;
% 
pred = [(1:testN)', yTe];
headers = {'ID', 'Prediction'};
csvwrite_with_headers('../data/svdpp_chop.csv',pred,headers);
