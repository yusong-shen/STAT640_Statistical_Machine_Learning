%% make Matrix Market input files

train = dlmread('../data/trainRatings.csv', ',', 1, 0);
validate = dlmread('../data/testRatings.csv', ',', 1, 0);
all = dlmread('../data/ratings.csv', ',', 1, 0);
idmap = dlmread('../data/IDMap.csv',',', 1, 0);

train = sparse(train(:,1), train(:,2), train(:,3));
validate = sparse(validate(:,1), validate(:,2), validate(:,3));
all = sparse(all(:,1), all(:,2), all(:,3));
idmap = sparse(idmap(:,1), idmap(:,2), idmap(:,3));


% mmwrite('../data/onlineDatingTrain/ratings.train', train);
% mmwrite('../data/onlineDatingTrain/ratings.validate', validate);
% mmwrite('../data/onlineDatingTest/all.train', all);
% mmwrite('../data/onlineDatingTest/idmap.validate', idmap);

mmwrite('/Users/yusong/Code/graphchi-cpp/ratings.train', train);
mmwrite('/Users/yusong/Code/graphchi-cpp/ratings.validate', validate);
mmwrite('/Users/yusong/Code/graphchi-cpp/all.train', all);
mmwrite('/Users/yusong/Code/graphchi-cpp/idmap.validate', idmap);
