
# Problem 1 : Digits data: 3 and 8

# use data processed by lecture's matlab code
# Todo : try to figure out how to use R to seperate 
zip_tr = read.csv("zip38_train.csv", header=FALSE );
zip_tr_labels = read.csv("zip38_train_labels.csv", header=FALSE );
zip_ts = read.csv("zip38_test.csv", header=FALSE);
zip_ts_labels = read.csv("zip38_test_labels.csv", header=FALSE);

train.feat = as.matrix(zip_tr)
train.resp = as.factor(as.matrix(zip_tr_labels))
test.feat = as.matrix(zip_ts)
test.resp = as.factor(as.matrix(zip_ts_labels))
n.train = length(train.resp)
n.test = length(test.resp)
# LDA
library(MASS)
lda.model <- lda(x=train.feat,grouping=train.resp)
lda.model$prior
lda.preds <- predict(lda.model,newdata = test.feat)
head(lda.preds$posterior)
lda.con.mat <- table(lda.preds$class,test.resp)
lda.con.mat
lda.test.err <- 1 - ( sum(diag(lda.con.mat)) / sum(lda.con.mat) )
lda.test.err


# QDA
# Error in qda.default(x, grouping, ...) : rank deficiency in group 8
# Need to jitter the data to avoid exact multicolinearity
train.feat.j <- train.feat
train.feat.j[,-1] <- apply(train.feat[,-1], 2, jitter)

qda.model <- qda(x=train.feat.j,grouping=train.resp)
qda.model$prior
# predict on test set
qda.preds <- predict(qda.model,newdata = test.feat)
head(qda.preds$posterior)
qda.con.mat <- table(qda.preds$class,test.resp)
qda.test.err <- 1 - ( sum(diag(qda.con.mat))  / sum(qda.con.mat) )
qda.test.err


# Naive Bayes
library(e1071)
nb.model <- naiveBayes(x=train.feat,y=train.resp)
nb.model$apriori / (n.train)
nb.preds <- predict(nb.model,newdata=test.feat)
nb.con.mat <- table(nb.preds,test.resp)
nb.con.mat
nb.test.err <- 1 - (sum(diag(nb.con.mat)) / sum(nb.con.mat))
nb.test.err

# KNN
knn.model <- tune.knn(train.feat,train.resp,k=1:100)
knn.model
plot(knn.model)
library(class)
knn.best <- knn(train=train.feat,test=test.feat,cl=train.resp,k=knn.model$best.parameters)
knn.con.mat <- table(knn.best,test.resp)
knn.con.mat
knn.test.err <- 1 - ( sum(diag(knn.con.mat))/sum(knn.con.mat))
knn.test.err



# Linear SVMs
library(kernlab)
linear.svm <- ksvm(x=train.feat, y=train.resp, kernel='vanilladot', 
                   type='C-svc', cross=10)
linear.svm
length(linear.svm@alphaindex[[1]])

lsvm.preds <- predict(linear.svm, newdata=test.feat)
lsvm.con.mat <- table(lsvm.preds,test.resp)
lsvm.con.mat
lsvm.test.err <- 1 - (sum(diag(lsvm.con.mat))) / sum(lsvm.con.mat)
lsvm.test.err


# Radial Kernels SVMs
radial.svm <- ksvm(x=train.feat, y=train.resp, kernel='rbfdot', 
                   type='C-svc', cross=10)
radial.svm
length(radial.svm@alphaindex[[1]])

rsvm.preds <- predict(radial.svm, newdata=test.feat)
rsvm.con.mat <- table(rsvm.preds,test.resp)
rsvm.con.mat
rsvm.test.err <- 1 - (sum(diag(rsvm.con.mat))) / sum(rsvm.con.mat)
rsvm.test.err

# Polynomial Kernels SVMs
poly.svm <- ksvm(x=train.feat, y=train.resp, kernel='polydot', 
                   kpar=list(degree=2, scale=1, offset=0), cross=10)
poly.svm
length(poly.svm@alphaindex[[1]])

psvm.preds <- predict(poly.svm, newdata=test.feat)
psvm.con.mat <- table(psvm.preds,test.resp)
psvm.con.mat
psvm.test.err <- 1 - (sum(diag(psvm.con.mat))) / sum(psvm.con.mat)
psvm.test.err


# # Logistic Regression
# wrap the training data into a data frame
train.feat.df <- as.data.frame(train.feat)
test.feat.df <- as.data.frame(test.feat)

library(nnet)

logit <- multinom(train.resp ~., data=train.feat.df, MaxNWts = 3000)
summary(logit)

# use lr.preict helper function
dump("lr.predict", file = "lr.redict.R")
source("lr.predict.R")
# Find training and test error rates
# Todo : prediction ??
logit.preds.train <- lr.predict(logit, train.feat.df)
logit.preds.test <- lr.predict(logit, test.feat.df)
logit.error <-  mean(logit.preds.test != test.resp)

# What does logistic regression mix up the most?
table(logit.preds.test, test.resp)

# Logistic Regression with regularization
# Todo

