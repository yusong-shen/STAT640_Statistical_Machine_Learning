% simple matrix factorization test

% set parameter
steps = 5000;
alpha = 0.0002;
beta = 0.02;

%
R = [ 5 3 0 1;
      4 0 0 1;
      1 1 0 5;
      0 1 5 4;
      ];
 N = size(R,1);
 M = size(R,2);
 K = 2;
 
 P = randi(5,N,K);
 Q = randi(5,M,K);
  
[Pfinal, Qfinal ] = ...
    matrixFactorization( R, P, Q, K, steps, alpha, beta );

Rpred = Pfinal*Qfinal';

