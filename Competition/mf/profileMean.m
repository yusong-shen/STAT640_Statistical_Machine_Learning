function [ pred ] = profileMean( rmat, idmap )
%profileMean Summary of this function goes here
%   Detailed explanation goes here


% row : user
% col : profile
Pnum = sum(rmat~=0,1);
Psum = sum(rmat,1);
Pmeans = Psum ./ Pnum;

Pred = Pmeans(idmap(:,2));
pred = [idmap(:,3),Pred'];


end

