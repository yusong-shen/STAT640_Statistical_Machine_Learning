%% Patition Dataset

% split the dataset 3,279,759 items
% into training set and test set

clear all;

ratings = dlmread('ratings.csv', ',', 1, 0);
n = size(ratings,1);

[trainInd,valInd,testInd] = divideRand(n,0.8,0,0.2);
trainSet = ratings(trainInd,:);
testSet = ratings(testInd,:);
% write to csv
% header1 = {'UserID', 'ProfileID', 'Rating'};
% csvwrite_with_headers('trainRatings.csv',trainSet,header1);
% csvwrite_with_headers('testRatings.csv',testSet,header1);

trainRmat = sparse(trainSet(:,1),trainSet(:,2),trainSet(:,3));
testRmat = sparse(testSet(:,1),testSet(:,2), testSet(:,3));


% sort by first column - UserID
testN = size(testSet,1);
testIDmap = sortrows([testSet(:,1),testSet(:,2)]);
testIDmap = [testIDmap, (1:testN)'];
% groundTruth 
groundTruth = [(1:testN)',zeros(testN,1)];
for i = 1:testN
    groundTruth(i,2) = testRmat(testIDmap(i,1),testIDmap(i,2));

end
% 
% % write to csv
% header2 = {'UserID', 'ProfileID', 'KaggleID'};
% header3 = {'ID', 'Prediction'};
% csvwrite_with_headers('testIDmap.csv', testIDmap, header2);
% csvwrite_with_headers('groundTruth.csv', groundTruth, header3);


%% Profile Mean
[ pred ] = profileMean( rmat, idmap );
pmRMSE = rmse(groundTruth(:,2),pred(:,2));
