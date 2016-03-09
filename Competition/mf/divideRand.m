function [trainInd,valInd,testInd] = divideRand(Q,trainRatio,valRatio,testRatio)
%divideRand Divide targets into three sets using random indices
%   [trainInd,valInd,testInd] = divideRand(3000,0.6,0.2,0.2);

trainN = int32(Q*trainRatio);
valN = int32(Q*valRatio);


rInd = randperm(Q);
trainInd = rInd(1:trainN);
valInd = rInd(trainN+1:trainN+valN);
testInd = rInd(trainN+1+valN:end);

end

