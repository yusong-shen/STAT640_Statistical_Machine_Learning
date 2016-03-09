##########
#Biclustering - Cluster Heatmap 

require("ISLR")
# ncidat = t(NCI60$data)
# colnames(ncidat) = NCI60$labs
train = read.csv("/Users/yusong/Code/STAT640/data/author_training.csv")
test = read.csv("/Users/yusong/Code/STAT640/data/author_testing.csv")
train_dat = train
test_dat = test
dat = rbind(train_dat, test_dat)
authors = dat[,70]
dat = dat[,1:69]

#filter genes using PCA
X = t(scale(t(dat),center=TRUE,scale=FALSE))
sv = svd(t(X));
V = sv$v

#PC loadings - visualize data by limiting to top genes in magnitude in the PC loadings 
aa = grep("grey",colors())
bb = grep("green",colors())
cc = grep("red",colors())
gcol2 = colors()[c(aa[1:30],bb[1:20],rep(cc,2))]

j = 2
ord = order(abs(V[,j]),decreasing=TRUE)
x = as.matrix(X[ord[1:250],])

#cluster heatmap - uses Ward's linkage (complete is default)
heatmap(x,col=gcol2,hclustfun=function(x)hclust(x,method="ward.D"))

######################################################

