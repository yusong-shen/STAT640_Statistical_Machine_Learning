%% STAT640 Competition
% Matlab script for Profile Mean Benchmark

% M = csvread('FILENAME',R,C)
% ratings = csvread('ratings.csv',1, 0);

clear all;

ratings = dlmread('../data/trainRatings.csv', ',', 1, 0);
idmap = dlmread('../data/testIDmap.csv',',', 1, 0);
rmat = sparse(ratings(:,1),ratings(:,2),ratings(:,3));
groundTruth = dlmread('../data/groundTruth.csv', ',', 1, 0);
% dlmwrite('../data/trainRmat.csv',full(rmat),'delimiter',',');

% row : user
% col : profile
Pnum = sum(rmat~=0,1);
Psum = sum(rmat,1);
Pmeans = Psum ./ Pnum;

Pred = Pmeans(idmap(:,2));
MatlabBenchmark = [idmap(:,3),Pred'];

pmRMSE = rmse(groundTruth(:,2),MatlabBenchmark(:,2));

% headers = {'ID', 'Prediction'};
% csvwrite_with_headers('MatlabBenchmark2.csv',full(MatlabBenchmark),headers);


