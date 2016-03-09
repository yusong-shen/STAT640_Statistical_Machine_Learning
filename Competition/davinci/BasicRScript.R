###############################
#script for Profile Mean Benchmark

require("Matrix")

ratings = read.table("/projects/stat640/Fall2015_Data/ratings.csv",header=TRUE,sep=",")
idmap = read.table("/projects/stat640/Fall2015_Data/IDMap.csv",header=TRUE,sep=",")

rmat = sparseMatrix(i=ratings[,1],j=ratings[,2],x=ratings[,3])

Pnum = apply(rmat!=0,2,sum)
Psum = apply(rmat,2,sum)
Pmeans = Psum / Pnum
# Print some output
head(Pmeans)

Pred = Pmeans[idmap[,2]]
head(Pred)

PMbenchmark = cbind(idmap[,3],Pred)
head(PMbenchmark)
colnames(PMbenchmark) = c("ID","Prediction")
write.table(PMbenchmark,file="/home/jn13/PMbenchmark.csv",quote=FALSE,sep=",",row.names=FALSE,col.names=TRUE)

