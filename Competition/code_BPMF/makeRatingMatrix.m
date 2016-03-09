%% Create a matrix of size num_p by num_m from triplets {user_id, movie_id, rating_id}  

% Triplets: {user_id, profile_id, rating} 
train_vec = dlmread('../data/trainRatings.csv', ',', 1, 0); 
probe_vec = dlmread('../data/testRatings.csv', ',', 1, 0); 

num_m = 10000;
num_p = 10000;
count = zeros(num_p,num_m,'single'); %for Netflida data, use sparse matrix instead. 

for mm=1:num_m
 ff= find(train_vec(:,2)==mm);
 count(train_vec(ff,1),mm) = train_vec(ff,3);
end 


