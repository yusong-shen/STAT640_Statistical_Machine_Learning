function [ Pfinal, Qfinal ] = matrixFactorization( R, P, Q, K, steps, alpha, beta )
%matrixFactorization Summary of this function goes here
%   Detailed explanation goes here
Q = Q';
for step = 1:steps
    fprintf('steps %d... \n',step);
    for i = 1:size(R,1)
        for j = 1:size(R,2)
            if R(i,j) > 0
                eij = R(i,j) - P(i,:)*Q(:,j);
                for k = 1:K
                    P(i,k) = P(i,k) + alpha*(2*eij*Q(k,j)-beta*P(i,k));
                    Q(k,j) = Q(k,j) + alpha*(2*eij*P(i,k)-beta*Q(k,j)); 
                end
            end
        end
    end
%     eR = P*Q;
    e = 0;   
    for i = 1:size(R,1)
        for j = 1:size(R,2)
            if R(i,j) > 0
                e = e + (R(i,j) - P(i,:)*Q(:,j)).^2;
                for k = 1:K
                    e = e + (beta/2) * (P(i,k).^2+Q(k,j).^2);
                end
            end
        end
        
    end
    fprintf('error %f \n',e);
    if e < 0.001
        break
    end        
end

Pfinal = P;
Qfinal = Q';


end

