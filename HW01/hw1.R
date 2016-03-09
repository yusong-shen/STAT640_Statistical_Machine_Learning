# For Stat 640 HW01
# 
# Gene Expression Data
# SRBCT microarray data
# n = 83 patients, p = 2308 genes
# Your response is the expression profile of the gene p53,
# a major oncongene that acts as a tumor suppressor.
# Your goal is to select other genes whoes expression profiles
# are associated with p53


#libraries required
library(glmnet)
library(ncvreg)

# Visualize regularization paths for:
# 1) Elastic net
# 2) Lasso
# 3) SCAD
# 4) MC+

# read the SRBCT genes data
# notice : R ignore the first line of csv file !!
X_SRBCT = read.csv("X_SRBCT_2.csv")
Y_SRBCT = read.csv("Y_SRBCT_2.csv")

Y = as.matrix(Y_SRBCT);Y = as.numeric(Y); Y = Y - mean(Y)
X = as.matrix(X_SRBCT); X = scale(X,center=T,scale=F)

# Lasso
lam = 1
fit0 = lm(Y~X-1)
fitl = glmnet(x=X,y=Y,family="gaussian",lambda=lam,alpha=1)
cbind(fit0$cief, as.matrix(fitl$beta))

# Lasso Path
fitl = glmnet(x=X,y=Y,family="gaussian",alpha=1)
plot(fitl)

###############################
#least squares, lasso, adaptive lasso, SCAD, ridge, elastic net, MC+
lam = 1

# betals = solve(t(X)%*%X)%*%t(X)%*%Y
# betar = solve(t(X)%*%X + diag(rep(lam/2*(nrow(X_SRBCT)+1),2307)))%*%t(X)%*%Y
fitl = glmnet(x=X,y=Y,family="gaussian",lambda=lam,alpha=1)
# fital = glmnet(x=X,y=Y,family="gaussian",lambda=lam,alpha=1,penalty.factor=1/abs(betals))
fitel = glmnet(x=X,y=Y,family="gaussian",lambda=lam,alpha=.5)
fitscad = ncvreg(X,Y,family="gaussian",penalty="SCAD",lambda=lam)
fitmcp = ncvreg(X,Y,family="gaussian",penalty="MCP",lambda=lam)
# mat = cbind(betals,betar,as.matrix(fitl$beta),as.matrix(fital$beta),as.matrix(fitel$beta),fitscad$beta[-1],fitmcp$beta[-1])
# colnames(mat) = c("LS","Ridge","Lasso","A-Lasso","EL","SCAD","MC+")
# mat

#############################
#compare ridge, lasso, elastic net & SCAD regualrization paths

par(mfrow=c(2,3))
par(mar=c(5,4,3,2))

fitl = glmnet(x=X,y=Y,family="gaussian",alpha=1)
betaLa = as.matrix(fitl$beta)
betaLaSum = apply(betaLa,1,sum)
laTop10 = head(sort(betaLaSum,decreasing=TRUE), 10)

# write.csv(as.matrix(fitl$beta),"lassoBetas.csv")
plot(fitl,main="Lasso")
# legend(0,20,legend=names(ozone)[2:9],col=1:8,lty=rep(1,8),cex=.75)


fitel = glmnet(x=X,y=Y,family="gaussian",alpha=.5)
betaEl50 = as.matrix(fitel$beta)
betaEl50Sum = apply(betaEl50,1,sum)
El50Top10 = head(sort(betaEl50Sum,decreasing=TRUE), 10)

# write.csv(as.matrix(fitel$beta),"elstic50Beta.csv")
plot(fitel,main="EL alpha=.5")
# legend(0,20,legend=names(ozone)[2:9],col=1:8,lty=rep(1,8),cex=.75)

fitel = glmnet(x=X,y=Y,family="gaussian",alpha=.25)
betaEl25 = as.matrix(fitel$beta)
betaEl25Sum = apply(betaEl25,1,sum)
El25Top10 = head(sort(betaEl25Sum,decreasing=TRUE), 10)
# write.csv(as.matrix(fitel$beta),"elstic25Beta.csv")
plot(fitel,main="EL alpha=.25")
# legend(0,20,legend=names(ozone)[2:9],col=1:8,lty=rep(1,8),cex=.75)

fitscad = ncvreg(X,Y,family="gaussian",penalty="SCAD")
betaSCAD = as.matrix(fitscad$beta)
betaSCADSum = apply(betaSCAD,1,sum)
SCADTop10 = head(sort(betaSCADSum,decreasing=TRUE), 10)
# write.csv(as.matrix(fitscad$beta),"scadBeta.csv")
plot(fitscad,main="SCAD",shade=T)
# legend(6,30,legend=names(ozone)[2:9],col=1:8,lty=rep(1,8),cex=.75)

fitmcp = ncvreg(X,Y,family="gaussian",penalty="MCP")
betaMCP = as.matrix(fitmcp$beta)
betaMCPSum = apply(betaMCP,1,sum)
MCPTop10 = head(sort(betaMCPSum,decreasing=TRUE), 10)
# write.csv(as.matrix(fitmcp$beta),"mcpBeta.csv")
plot(fitmcp,main="MC+",shade=T)
# legend(6,30,legend=names(ozone)[2:9],col=1:8,lty=rep(1,8),cex=.75)

# Print Top 10
laTop10
El50Top10
El25Top10
SCADTop10
MCPTop10