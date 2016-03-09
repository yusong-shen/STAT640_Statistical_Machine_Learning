%%  Add the gender as time slice

tr_ratings = dlmread('../data/trainRatings.csv', ',', 1, 0);
ts_ratings = dlmread('../data/testRatings.csv', ',', 1, 0);
idmap = dlmread('../data/IDMap.csv',',', 1, 0);
gender = dlmread('../data/gender.csv', ',', 1, 0);

n_tr = size(tr_ratings,1);
n_ts = size(ts_ratings,1);

TTr.subs = [tr_ratings(:,1), tr_ratings(:,2), ones(n_tr,1)];
TTe.subs = [ts_ratings(:,1), ts_ratings(:,2), ones(n_ts,1)];

for i = 1:n_tr
    TTr.subs(i,3) = gender(tr_ratings(i,1),2);
end

for i = 1:n_ts
    TTe.subs(i,3) = gender(ts_ratings(i,1),2);
end

TTr.vals = tr_ratings(:,3);
TTe.vals = ts_ratings(:,3);

TTr.size = [10000, 10000, 3];
TTe.size = [10000, 10000, 3];



save TTr TTr
save TTe TTe


%%
n_id = size(idmap, 1);
idmap_kaggle = [idmap(:,1:2), ones(n_id,1)];
for i = 1:n_id
    idmap_kaggle(i,3) = gender(idmap_kaggle(i,1),2);
end

save idmap idmap_kaggle

