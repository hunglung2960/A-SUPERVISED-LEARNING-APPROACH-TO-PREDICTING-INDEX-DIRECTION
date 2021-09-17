rm(list=ls())
setwd("C://Users/Patrick/Desktop/MLProject/Data")

#######################
##     Library       ##
#######################
library(readxl)
library(tseries)
library(HDeconometrics)
library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)
library(hdm)
library(ggplot2)
library(gridExtra)
library(dplyr)
theme_set(theme_minimal())


#######################
##     Function      ##
#######################
## This function forming forecasts using classification tree with no cp, 1SE rule or min CV
runtree=function(Y,indice,lag,cp){
  Y2 = Y
  aux=embed(Y2,4+lag)
  y=aux[,indice]
  X=aux[,-c(1:(ncol(Y2)*lag))]
  
  if(lag==1){
    X.out=tail(aux,1)[1:ncol(X)]
  }else{
    X.out=aux[,-c(1:(ncol(Y2)*(lag-1)))]
    X.out=tail(X.out,1)[1:ncol(X)]
  }
  
  df <- as.data.frame(cbind(X,y))
  treeGini= rpart(y~., data=df, method = "class", minsplit = 10, cp = .0001, maxdepth = 30)
  
  SEcpGini=treeGini$cptable[2]
  bestcpGini=treeGini$cptable[which.min(treeGini$cptable[,"xerror"]),"CP"]
  
  model0 = prune(treeGini,cp=SEcpGini)
  model1 = prune(treeGini,cp=bestcpGini)
  model2 = rpart(y~., data=df, method = "class", minsplit = 10, maxdepth = 30)
  
  X.out.df <- as.data.frame(matrix(X.out, nrow = 1))
  pred0 = predict(model0,newdata = X.out.df)[,2]
  size0 = length(unique(model0$where))
  importance0 = model0$variable.importance
  pred1 = predict(model1,newdata = X.out.df)[,2]
  size1 = length(unique(model1$where))
  importance1 = model1$variable.importance
  pred2 = predict(model2,newdata = X.out.df)[,2]
  size2 = length(unique(model2$where))
  importance2 = model2$variable.importance
  
  if (cp==0) {
    return(list("model"=model0,"pred"=pred0, "size"=size0, "importance" = importance0))
  }
  else if (cp==2) {
    return(list("model"=model1,"pred"=pred1, "size"=size1, "importance" = importance1))
  }
  else {
    return(list("model"=model2,"pred"=pred2, "size"=size2, "importance" = importance2))
  }
}


## This function will repeatedly call the previous function in the expanding window h-step forecasting
# cp=0 to use 1SE, cp=1 to use min CV, cp=2 to use no cp
tree.expanding.window=function(Y,nprev,indice=1,lag=1,cp){
  save.size=c()
  save.pred=matrix(NA,nprev,1)
  save.importance = list()
  for(i in nprev:1){
    Y.window=Y[1:(nrow(Y)-i),]
    lasso=runtree(Y.window,indice,lag,cp)
    save.pred[(1+nprev-i),]=lasso$pred
    save.size[(1+nprev-i)]=lasso$size
    save.importance[[(1+nprev-i)]]=lasso$importance
    cat("iteration",(1+nprev-i),"\n")
  }
  
  return(list("pred"=save.pred, "size"=save.size, "importance"=save.importance))
}


## This function forming forecasts using bagging or random forests
runrf=function(Y,indice,lag, rf){
  Y2 = Y
  aux=embed(Y2,4+lag)
  y=aux[,indice]
  X=aux[,-c(1:(ncol(Y2)*lag))] 
  
  if(lag==1){
    X.out=tail(aux,1)[1:ncol(X)]
  }else{
    X.out=aux[,-c(1:(ncol(Y2)*(lag-1)))]
    X.out=tail(X.out,1)[1:ncol(X)]
  }
  
  if (rf==0) {
    mtrytem=ncol(X)
  }
  else {
    mtrytem=sqrt(ncol(X))
  }
  
  model=randomForest(X,as.factor(y),importance=TRUE, mtry=mtrytem)
  pred=as.numeric(as.character(predict(model,X.out)))
  prob=as.data.frame(predict(model, X.out, type="prob"))
  
  return(list("model"=model,"pred"=pred, "prob"=prob))
}


## This function will repeatedly call the previous function in the expanding window h-step forecasting
# rf=0 if bagging, rf=1 if rf
rf.expanding.window=function(Y,nprev,indice=1,lag=1, rf){ 
  save.importance=list()
  save.pred=matrix(NA,nprev,1)
  save.prob = data.frame(matrix(nrow=nprev,ncol=0))
  for(i in nprev:1){
    Y.window=Y[1:(nrow(Y)-i),]
    lasso=runrf(Y.window,indice,lag,rf)
    save.pred[(1+nprev-i),]=lasso$pred
    save.importance[[i]]=importance(lasso$model)
    save.prob = rbind(save.prob, lasso$prob)
    cat("iteration",(1+nprev-i),"\n")
  }
  
  return(list("pred"=save.pred,"save.importance"=save.importance, "prob"=save.prob))
}


## This function forming forecasts using boosting with cv or without cv
runb=function(Y,indice,lag,cv){
  Y2 = Y
  aux=embed(Y2,4+lag)
  y=aux[,indice]
  X=aux[,-c(1:(ncol(Y2)*lag))]
  
  if(lag==1){
    X.out=tail(aux,1)[1:ncol(X)]
  }else{
    X.out=aux[,-c(1:(ncol(Y2)*(lag-1)))]
    X.out=tail(X.out,1)[1:ncol(X)]
  }
  
  df <- as.data.frame(cbind(X,y))
  if (cv==1) {
    model = gbm(y~.,data=df,distribution='bernoulli',
                interaction.depth=5,n.trees=1000,shrinkage=0.01, cv.folds = 10)
  }
  else {
    model = gbm(y~.,data=df,distribution='bernoulli', interaction.depth=5,shrinkage=0.01)
  }
  
  
  bestcv = gbm.perf(model, method="cv")
  
  X.out.df <- as.data.frame(matrix(X.out, nrow = 1))
  pred = predict(model,newdata = X.out.df,n.trees = bestcv, type = "response")
  
  vimp = summary(model, n.trees = bestcv, plotit = FALSE)
  
  return(list("model"=model,"pred"=pred,"vimp"=vimp))
}


## This function will repeatedly call the previous function in the expanding window h-step forecasting
# cv=1 if with cv, cv=0 if without cv
boosting.expanding.window=function(Y,nprev,indice=1,lag=1,cv){ 
  save.importance=list()
  save.pred=matrix(NA,nprev,1)
  for(i in nprev:1){
    Y.window=Y[1:(nrow(Y)-i),]
    lasso=runb(Y.window,indice,lag,cv)
    save.pred[(1+nprev-i),]=lasso$pred
    save.importance[[i]]=lasso$vimp
    cat("iteration",(1+nprev-i),"\n")
  }
  
  return(list("pred"=save.pred,"save.importance"=save.importance))
}


## This function forming forecasts using logistic LASSO
runll=function(Y,indice,lag){
  Y2 = Y
  aux=embed(Y2,4+lag)
  y=aux[,indice]
  X=aux[,-c(1:(ncol(Y2)*lag))]
  
  if(lag==1){
    X.out=tail(aux,1)[1:ncol(X)]
  }else{
    X.out=aux[,-c(1:(ncol(Y2)*(lag-1)))]
    X.out=tail(X.out,1)[1:ncol(X)]
  }
  
  model = rlassologit(y~X,  post=FALSE)
  
  newX <- matrix(X.out, nrow = 1)
  pred<- predict(model, newdata=newX)
  
  numselected <- sum(model$index)
  selected <- which(model$index)
  
  return(list("model"=model,"pred"=pred, "numselected"=numselected, "selected"=selected))
}


## This function will repeatedly call the previous function in the expanding window h-step forecasting
lassologit.expanding.window=function(Y,nprev,indice=1,lag=1){
  
  save.numselected=c()
  save.selected=list()
  save.pred=matrix(NA,nprev,1)
  for(i in nprev:1){
    Y.window=Y[1:(nrow(Y)-i),]
    lasso=runll(Y.window,indice,lag)
    save.pred[(1+nprev-i),]=lasso$pred
    save.numselected[(1+nprev-i)]=lasso$numselected
    save.selected[[(1+nprev-i)]]=lasso$selected
    cat("iteration",(1+nprev-i),"\n")
  }
  
  return(list("pred"=save.pred, "numselected"=save.numselected, "selected"=save.selected))
}


## This function forming forecasts using linear model
runum=function(Y,indice,lag){
  Y2 = Y
  aux=embed(Y2,4+lag)
  y=aux[,indice]
  X=aux[,-c(1:(ncol(Y2)*lag))]
  
  if(lag==1){
    X.out=tail(aux,1)[1:ncol(X)]
  }else{
    X.out=aux[,-c(1:(ncol(Y2)*(lag-1)))]
    X.out=tail(X.out,1)[1:ncol(X)]
  }

  model = lm(y~1)
  pred<- predict(model)[1]
  
  return(list("model"=model,"pred"=pred))
}


## This function will repeatedly call the previous function in the expanding window h-step forecasting
um.expanding.window=function(Y,nprev,indice=1,lag=1){
  
  save.pred=matrix(NA,nprev,1)
  for(i in nprev:1){
    Y.window=Y[1:(nrow(Y)-i),]
    lasso=runum(Y.window,indice,lag)
    save.pred[(1+nprev-i),]=lasso$pred
    cat("iteration",(1+nprev-i),"\n")
  }
  
  return(list("pred"=save.pred))
}


## This function forming forecasts using probit model
runpm=function(Y,indice,lag){
  Y2 = Y
  aux=embed(Y2,4+lag)
  y=aux[,indice]
  X=aux[,-c(1:(ncol(Y2)*lag))]
  
  if(lag==1){
    X.out=tail(aux,1)[1:ncol(X)]
  }else{
    X.out=aux[,-c(1:(ncol(Y2)*(lag-1)))]
    X.out=tail(X.out,1)[1:ncol(X)]
  }
  
  model = glm(y~X,family = binomial(link = "probit"))
  
  tem = model$coefficients[1]
  tem1 = model$coefficients[-1]
  tem1[is.na(tem1)]=0
  pred = matrix(tem1,nrow=1)%*%matrix(X.out,ncol=1)+tem
  pred = pnorm(pred)
  
  return(list("model"=model,"pred"=pred))
}


## This function will repeatedly call the previous function in the expanding window h-step forecasting
pm.expanding.window=function(Y,nprev,indice=1,lag=1){
  save.pred=matrix(NA,nprev,1) 
  for(i in nprev:1){
    Y.window=Y[1:(nrow(Y)-i),]
    lasso=runpm(Y.window,indice,lag)
    save.pred[(1+nprev-i),]=lasso$pred
    cat("iteration",(1+nprev-i),"\n")
  }
  
  return(list("pred"=save.pred))
}


## This function plot variable importance for classification tree
t.impt.plot=function(rftem) {
  pindex <- as.numeric(sub(".","",names(unlist(rftem$importance))))
  
  pname <- colnames(Y)[ifelse(pindex %% 101 == 0 , 101,pindex %% 101)]
  plotdf0 <- data.frame(pname = pname, pvalue = unlist(rftem$importance))
  plotdf_final0 <- plotdf0 %>% group_by(pname) %>% summarise(pvalue = sum(pvalue)) %>% arrange(-pvalue)
  plot1 <- plotdf_final0[1:10,]
  
  nonlinear_cond <- !(grepl(".square", plotdf_final0$pname) | grepl(".cubic", plotdf_final0$pname))
  plot2 <- plotdf_final0[nonlinear_cond,][1:10,]
  
  one <- ggplot(data = plot1) + 
    geom_bar(aes(y=reorder(pname, pvalue), x= pvalue),
             stat="identity", fill ="steelblue") +
    theme(axis.title.x=element_blank(), axis.text.x=element_blank(), 
          axis.ticks.x=element_blank()) +
    theme(axis.title.y=element_blank(), axis.text.y=element_text(colour="black")) +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank())
  
  two <- ggplot(data = plot2) + 
    geom_bar(aes(y=reorder(pname, pvalue), x= pvalue),
             stat="identity", fill ="steelblue") +
    theme(axis.title.x=element_blank(), axis.text.x=element_blank(), 
          axis.ticks.x=element_blank()) +
    theme(axis.title.y=element_blank(), axis.text.y=element_text(colour="black")) +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank())
  
  return(grid.arrange(two,one, top ="Importance Measure (Excluding & Including Nonlinear Terms)"))
}


## This function plot variable importance for bagging and random forests
impt.plot=function(rftem,nprev) {
  varimpt <- matrix(0, nrow = 404, ncol = 2)
  for (i in 1:nprev) {
    varimpt <- varimpt + rftem$save.importance[[i]][,c(1,2)]
  }
  
  pvalue <- varimpt[,1][order(-varimpt[,1])]
  pname <- colnames(Y)[ifelse(order(-varimpt[,1]) %% 101 ==0,101,order(-varimpt[,1]) %% 101)]
  plotdf0 <- data.frame(pname = pname, pvalue = pvalue)
  plotdf_final0 <- plotdf0 %>% group_by(pname) %>% summarise(pvalue = sum(pvalue)) %>% arrange(-pvalue)
  plot1 <- plotdf_final0[1:10,]
  
  nonlinear_cond <- !(grepl(".square", plotdf_final0$pname) | grepl(".cubic", plotdf_final0$pname))
  plot2 <- plotdf_final0[nonlinear_cond,][1:10,]
  
  one <- ggplot(data = plot1) + 
    geom_bar(aes(y=reorder(pname, pvalue), x= pvalue),
             stat="identity", fill ="steelblue") +
    theme(axis.title.x=element_blank(), axis.text.x=element_blank(), 
          axis.ticks.x=element_blank()) +
    theme(axis.title.y=element_blank(), axis.text.y=element_text(colour="black")) +
    ggtitle("Importance Measure-0 (Including non-linear terms)")
  
  two <- ggplot(data = plot2) + 
    geom_bar(aes(y=reorder(pname, pvalue), x= pvalue),
             stat="identity", fill ="steelblue") +
    theme(axis.title.x=element_blank(), axis.text.x=element_blank(), 
          axis.ticks.x=element_blank()) +
    theme(axis.title.y=element_blank(), axis.text.y=element_text(colour="black")) +
    ggtitle("Importance Measure-0 (Exlcuding non-linear terms)")  
  
  pvalue <- varimpt[,2][order(-varimpt[,2])]
  pname <- colnames(Y)[ifelse(order(-varimpt[,2]) %% 101 ==0,101,order(-varimpt[,2]) %% 101)]
  plotdf1 <- data.frame(pname = pname, pvalue = pvalue)
  plotdf_final1 <- plotdf1 %>% group_by(pname) %>% summarise(pvalue = sum(pvalue)) %>% arrange(-pvalue)
  plot3 <- plotdf1[1:10,]
  
  nonlinear_cond <- !(grepl(".square", plotdf_final1$pname) | grepl(".cubic", plotdf_final1$pname))
  plot4 <- plotdf_final1[nonlinear_cond,][1:10,]
  
  three <- ggplot(data = plot3) + 
    geom_bar(aes(y=reorder(pname, pvalue), x= pvalue),
             stat="identity", fill ="steelblue") +
    theme(axis.title.x=element_blank(), axis.text.x=element_blank(), 
          axis.ticks.x=element_blank()) +
    theme(axis.title.y=element_blank(), axis.text.y=element_text(colour="black")) +
    ggtitle("Importance Measure-1 (Including non-linear terms)")
  
  four <- ggplot(data = plot4) + 
    geom_bar(aes(y=reorder(pname, pvalue), x= pvalue),
             stat="identity", fill ="steelblue") +
    theme(axis.title.x=element_blank(), axis.text.x=element_blank(), 
          axis.ticks.x=element_blank()) +
    theme(axis.title.y=element_blank(), axis.text.y=element_text(colour="black")) +
    ggtitle("Importance Measure-1 (Exlcuding non-linear terms)")  
  
  return(grid.arrange(two,one,four,three))
}


## This function plot variable importance for boosting
b.impt.plot=function(btem) {
  for (i in 1:length(btem$save.importance)) {
    pindex <- as.numeric(sub(".", "", btem$save.importance[[i]][,1]))
    btem$save.importance[[i]][,3] <- pindex
  }
  
  varimpt <- as.matrix(btem$save.importance[[1]][order(btem$save.importance[[1]][,3]),][,-c(1,3)])
  for (i in 2:length(btem$save.importance)) {
    varimpt <- varimpt + as.matrix(btem$save.importance[[i]][order(btem$save.importance[[i]][,3]),][,-c(1,3)])
  }
  
  rownames(varimpt) <- 1:nrow(varimpt)
  
  #Plotting
  pvalue <- varimpt[,1][order(-varimpt[,1])]
  pname <- colnames(Y)[ifelse(order(-varimpt[,1]) %% 101 ==0,101,order(-varimpt[,1]) %% 101)]
  plotdf0 <- data.frame(pname = pname, pvalue = pvalue)
  plotdf_final0 <- plotdf0 %>% group_by(pname) %>% summarise(pvalue = sum(pvalue)) %>% arrange(-pvalue)
  plot1 <- plotdf_final0[1:10,]
  
  nonlinear_cond <- !(grepl(".square", plotdf_final0$pname) | grepl(".cubic", plotdf_final0$pname))
  plot2 <- plotdf_final0[nonlinear_cond,][1:10,]
  
  one <- ggplot(data = plot1) + 
    geom_bar(aes(y=reorder(pname, pvalue), x= pvalue),
             stat="identity", fill ="steelblue") +
    theme(axis.title.x=element_blank(), axis.text.x=element_blank(), 
          axis.ticks.x=element_blank()) +
    theme(axis.title.y=element_blank(), axis.text.y=element_text(colour="black")) +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank())
  
  two <- ggplot(data = plot2) + 
    geom_bar(aes(y=reorder(pname, pvalue), x= pvalue),
             stat="identity", fill ="steelblue") +
    theme(axis.title.x=element_blank(), axis.text.x=element_blank(), 
          axis.ticks.x=element_blank()) +
    theme(axis.title.y=element_blank(), axis.text.y=element_text(colour="black")) +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank())
  
  grid.arrange(two,one, top ="Importance Measure (Excluding & Including Nonlinear Terms)")
}


## This function plot variable importance for logistic LASSO
ll.impt.plot=function(lltem) {
  count <- as.factor(unlist(lltem$selected))
  pcount <- summary(count)
  pnames <- as.numeric(names(summary(count)))
  actual <- colnames(Y)[ifelse(pnames %% 101 ==0, 101, pnames %% 101)]
  
  plot <- data.frame(name = actual, count = pcount)
  plotf <- plot %>% group_by(name) %>% summarize(count = sum(count)) %>% arrange(-count)
  
  ggplot(data = plotf) + geom_bar(aes(y=reorder(name, count), x= count),
                                  stat="identity", fill ="steelblue") +
    ylab("Selected Predictors") +
    xlab("Frequency")
}


## This function plot ROC curve
# tree=1 if classification tree and lassologit, tree=0 if else
roc=function(rftem, tree) { 
  ns=1000
  sv = seq(from=.0,to=.99,length.out=ns)
  
  FP = rep(0,ns)
  TP = rep(0,ns)
  N = rep(0,ns)
  n0 = sum(oosy==0)
  
  if (tree==0) {
    tem = rftem$prob[,2]
  }
  else {
    tem = rftem$pred
  }
  
  for(i in 1:ns) {
    N[i] = sum(tem>sv[i])/length(oosy)
    TP[i] = sum((tem>sv[i]) & (oosy==1))/sum(oosy==1)
    FP[i] = sum((tem>sv[i]) & (oosy==0))/sum(oosy==0)
  }
  
  par(mai=c(0.9,0.9,0.4,0.4))
  par(mfrow=c(1,2))
  plot(FP,TP,type='l',col='blue',cex.lab=2.0, ylab = 'TP', xlab = 'FP')
  abline(0,1,lty=2)
  title(main='ROC',cex.main=2)
  plot(N,TP,type='l',col='blue',cex.lab=2.0, ylab = 'TP', xlab = 'N')
  abline(0,1,lty=2)
  title(main='Lift',cex.main=2)
}


#######################
##       Data        ##
#######################
data = read_xlsx("PredictorData2019.xlsx", sheet="Monthly")
taidata = read.csv("taidata.csv", header=FALSE)
data2 = read.csv("FRED-MD Macro Data.csv", header=TRUE)


## Observe NaN
tem_name = names(data)[-1]
na = matrix(vector("character"),ncol=1,nrow=length(tem_name))
rownames(na) = tem_name
for (n in tem_name) {
  tem = data[n]=="NaN"
  tem[tem,] =1
  tem1 = c(0,tem)
  tem = c(tem,0)
  tem2 = tem-tem1
  if (sum(tem2!=0)==0) {
    next
  }
  na[n,] = paste(paste(which(tem2==1),which(tem2==-1)-1,sep=" to "),collapse=", ")
}
na # a table to show which observation is missing values

## Remove csp
data = data[!colnames(data)=="csp"]

## Remove NaN in data
for (n in names(data)) {
  data = data[data[n]!="NaN",]
  data[n] = as.numeric(c(data[n])[[1]])
}

## Filter predictors needed from data2
filter = regexpr("sasdate|M1SL|M2SL|M2REAL|INDPRO|IPFPNSS|IPFINAL|IPCONGD|IPDCONGD|
                 IPNCONGD|IPBUSEQ|IPMAT|IPDMAT|IPNMAT|IPMANSICS|IPB51222s|IPFUELS|NAPMPI|UNRATE", names(data2))
filter = attr(filter,"match.length")
data2 = data2[,filter!=-1]

code = data2[1,-1] # transformation code

## Remove Na in data2
data2 = data2[-1,]
data2 = na.omit(data2)

## Obeserve the range of date in data & data2
head(data$yyyymm,1)
tail(data$yyyymm,1)
head(data2$sasdate,1)
tail(data2$sasdate,1)

## Filter the range of data & data2
data = data[which(data$yyyymm==195901):nrow(data),]
data2 = data2[1:which(data2$sasdate=="12/1/2019"),]

## Merge data & data2
data = cbind(data, data2[,2:length(data2)])

## Convert continuous Index into binary
d = data$Index[2:nrow(data)]-data$Index[-nrow(data)]
d[d>=0] = 1
d[d<0] = 0
d = c(NA,d)
data$Index = d
data = data[-1,]

## Period: 1974.12 - 2015.12
data = data[which(data$yyyymm==197412):which(data$yyyymm==201512),]
data = cbind(data, taidata[c(-1,-2,-13)])
names(data)[33:42] = c("MAI_1_9_t-1", "MAI_1_12_t-1", "MAI_2_9_t-1", "MAI_2_12_t-1", 
                       "MOI_9_t-1", "MOI_12_t-1", "OBV_1_9_t-1", "OBV_1_12_t-1", "OBV_2_9_t-1", "OBV_2_12_t-1")


## Adding polynomial order 2 & order 3 for all variables
preds <- data[,c(-1,-2)]
tempsq <- cbind(preds[[1]]^2)
for (i in 2:ncol(preds)) { #Polynomial order 2
  tempsq <- cbind(tempsq, preds[[i]]^2)
}

colnames(tempsq) <- paste0(colnames(preds),".square")

tempcubic <- cbind(preds[[1]]^3)
for (i in 2:ncol(preds)) { #Polynomial order 3
  tempcubic <- cbind(tempcubic, preds[[i]]^3)
}

colnames(tempcubic) <- paste0(colnames(preds),".cubic")

data_final <- cbind(data,tempsq,tempcubic)

## Dropping duplicated columns (i.e. binary variables)
data_final <- unique.matrix(data_final, MARGIN=2)

## Checking for NA
sum(is.na(data_final))

## Transformation for FRED-MD variable (suggestion from FRED-MD)
for (i in 1:15) {
  no = code[i]
  n = names(no)
  if (code==2) {
    tem = data_final[n][2:nrow(data_final),]
    tem2 = data_final[n][-nrow(data_final),]
    tem3 = tem - tem2
    tem3 = c(NA, tem3)
    data_final[n] = tem3
    next
  }
  data_final[n] = log(data_final[n])
  tem = data_final[n][2:nrow(data_final),]
  tem2 = data_final[n][-nrow(data_final),]
  tem3 = tem - tem2
  tem3 = c(NA, tem3)
  data_final[n] = tem3
  if (code==6) {
    tem = data_final[n][3:nrow(data_final),]
    tem2 = data_final[n][c(-1,-nrow(data_final)),]
    tem3 = tem - tem2
    tem3 = c(NA,NA, tem3)
    data_final[n] = tem3
  }
}

## Transformation for other not binary variables
trend = c()
for (n in names(data_final)[c(-1:-2,-18:-42)]) {
  pvalue = adf.test(c(data_final[n])[[1]])$p.value
  if (pvalue>0.01) {
    trend = c(trend, n)
  }
}
trend # variables that do not reject null hypothesis (do not necassary non-stationary)

## Manually plot to check trend
for (n in trend) {
  plot(c(data_final[n])[[1]])
}

## Transforms variables by taking the difference
save.name = c()
for (n in trend) {
  tem = data_final[n][2:nrow(data_final),]
  tem2 = data_final[n][-nrow(data_final),]
  tem3 = tem - tem2
  
  #recheck
  pvalue = adf.test(tem3)$p.value
  if (pvalue>0.01) {
    save.name = c(save.name,n)
  }
  
  tem3 = c(NA, tem3)
  data_final[n] = tem3
}
save.name # variables that are still reject the test

## Manually plot to check trend
for (n in save.name) {
  plot(c(data_final[n])[[1]])
}

data_final=data_final[-1,] # remove NA casued by transformation

## Transforms variables by taking the difference
save2 = c()
for (n in save.name) {
  tem = data_final[n][2:nrow(data_final),]
  tem2 = data_final[n][-nrow(data_final),]
  tem3 = tem - tem2
  
  #recheck
  pvalue = adf.test(tem3)$p.value
  if (pvalue>0.01) {
    save2 = c(save2,n)
  }
  
  tem3 = c(NA, tem3)
  data_final[n] = tem3
}
save2 # variables that are still reject the test 

data_final=data_final[-1,]

## data_final from 1976.01 to 2015.12
data_final = data_final[which(data_final$yyyymm==197601):nrow(data_final),]

#write.csv(data_final, "data_final.csv")

####################################################################

impt.data = data_final[,1:42] # without non-linear variables
data=data_final

Y=as.matrix(data[,-1]) # matrix including response and all predictors
yy=Y[,1] # response
nprev=nrow(data) - which(data$yyyymm==200501) + 1 # number of test observations
oosy=tail(yy,nprev) # test response


#######################
# Classification tree #
#######################
## 1SE
set.seed(12345)
tree1c.1SE=tree.expanding.window(Y,nprev,1,1,0)
table((tree1c.1SE$pred > .5), tail(yy,nprev))
tree1c.1SE$size
roc(tree1c.1SE,1)
t.impt.plot(tree1c.1SE)

set.seed(12345)
tree3c.1SE=tree.expanding.window(Y,nprev,1,3,0)
table((tree3c.1SE$pred > .5), tail(yy,nprev))
tree3c.1SE$size
roc(tree3c.1SE,1)

set.seed(12345)
tree6c.1SE=tree.expanding.window(Y,nprev,1,6,0)
table((tree6c.1SE$pred > .5), tail(yy,nprev))
tree6c.1SE$size
roc(tree6c.1SE,1)

set.seed(12345)
tree12c.1SE=tree.expanding.window(Y,nprev,1,12,0)
table((tree12c.1SE$pred > .5), tail(yy,nprev))
tree12c.1SE$size
roc(tree12c.1SE,1)

## min CV
set.seed(12345)
tree1c.CV=tree.expanding.window(Y,nprev,1,1,1)
table((tree1c.CV$pred > .5), tail(yy,nprev))
tree1c.CV$size
roc(tree1c.CV,1)

set.seed(12345)
tree3c.CV=tree.expanding.window(Y,nprev,1,3,1)
table((tree3c.CV$pred > .5), tail(yy,nprev))
tree3c.CV$size
roc(tree3c.CV,1)

set.seed(12345)
tree6c.CV=tree.expanding.window(Y,nprev,1,6,1)
table((tree6c.CV$pred > .5), tail(yy,nprev))
tree6c.CV$size
roc(tree6c.CV,1)

set.seed(12345)
tree12c.CV=tree.expanding.window(Y,nprev,1,12,1)
table((tree12c.CV$pred > .5), tail(yy,nprev))
tree12c.CV$size
roc(tree12c.CV,1)

## no cp
set.seed(12345)
tree1c.no=tree.expanding.window(Y,nprev,1,1,2)
table((tree1c.no$pred > .5), tail(yy,nprev))
tree1c.no$size
roc(tree1c.no,1)

set.seed(12345)
tree3c.no=tree.expanding.window(Y,nprev,1,3,2)
table((tree3c.no$pred > .5), tail(yy,nprev))
tree3c.no$size
roc(tree3c.no,1)

set.seed(12345)
tree6c.no=tree.expanding.window(Y,nprev,1,6,2)
table((tree6c.no$pred > .5), tail(yy,nprev))
tree6c.no$size
roc(tree6c.no,1)

set.seed(12345)
tree12c.no=tree.expanding.window(Y,nprev,1,12,2)
table((tree12c.no$pred > .5), tail(yy,nprev))
tree12c.no$size
roc(tree12c.no,1)


#######################
#       Bagging       #
#######################
set.seed(12345)
bg1c = rf.expanding.window(Y,nprev,1,1,0)
table((bg1c$pred > .5), tail(yy,nprev))
impt.plot(bg1c,nprev)
roc(bg1c,0)

set.seed(12345)
bg3c=rf.expanding.window(Y,nprev,1,3,0)
table((bg3c$pred > .5), tail(yy,nprev))
impt.plot(bg3c,nprev)
roc(bg3c,0)

set.seed(12345)
bg6c=rf.expanding.window(Y,nprev,1,6,0)
table((bg6c$pred > .5), tail(yy,nprev))
impt.plot(bg6c,nprev)
roc(bg6c,0)

set.seed(12345)
bg12c=rf.expanding.window(Y,nprev,1,12,0)
table((bg12c$pred > .5), tail(yy,nprev))
impt.plot(bg12c,nprev)
roc(bg12c,0)


#######################
#         RF          #
#######################
set.seed(12345)
rf1c = rf.expanding.window(Y,nprev,1,1,1)
table((rf1c$pred > .5), tail(yy,nprev))
impt.plot(rf1c,nprev)
roc(rf1c,0)

set.seed(12345)
rf3c=rf.expanding.window(Y,nprev,1,3,1)
table((rf3c$pred > .5), tail(yy,nprev))
impt.plot(rf3c,nprev)
roc(rf3c,0)

set.seed(12345)
rf6c=rf.expanding.window(Y,nprev,1,6,1)
table((rf6c$pred > .5), tail(yy,nprev))
impt.plot(rf6c,nprev)
roc(rf6c,0)

set.seed(12345)
rf12c=rf.expanding.window(Y,nprev,1,12,1)
table((rf12c$pred > .5), tail(yy,nprev))
impt.plot(rf12c,nprev)
roc(rf12c,0)


#######################
#      Boosting       #
#######################
## with cv
set.seed(12345)
b1c.cv=boosting.expanding.window(Y,nprev,1,1,1)
table((b1c.cv$pred > .5), tail(yy,nprev))
b.impt.plot(b1c.cv)
roc(b1c.cv,1)

set.seed(12345)
b3c.cv=boosting.expanding.window(Y,nprev,1,3,1)
table((b3c.cv$pred > .5), tail(yy,nprev))
b.impt.plot(b3c.cv)
roc(b3c.cv,1)

set.seed(12345)
b6c.cv=boosting.expanding.window(Y,nprev,1,6,1)
table((b6c.cv$pred > .5), tail(yy,nprev))
b.impt.plot(b6c.cv)
roc(b6c.cv,1)

set.seed(12345)
b12c.cv=boosting.expanding.window(Y,nprev,1,12,1)
table((b12c.cv$pred > .5), tail(yy,nprev))
b.impt.plot(b12c.cv)
roc(b12c.cv,1)

## without cv
b1c.ncv=boosting.expanding.window(Y,nprev,1,1,1)
table((b1c.ncv$pred > .5), tail(yy,nprev))
b.impt.plot(b1c.ncv)

set.seed(12345)
b3c.ncv=boosting.expanding.window(Y,nprev,1,3,1)
table((b3c.ncv$pred > .5), tail(yy,nprev))
b.impt.plot(b3c.ncv)

set.seed(12345)
b6c.ncv=boosting.expanding.window(Y,nprev,1,6,1)
table((b6c.ncv$pred > .5), tail(yy,nprev))
b.impt.plot(b6c.ncv)

set.seed(12345)
b12c.ncv=boosting.expanding.window(Y,nprev,1,12,1)
table((b12c.ncv$pred > .5), tail(yy,nprev))
b.impt.plot(b12c.ncv)


#######################
#     Lassologit      #
#######################
set.seed(12345)
ll1c=lassologit.expanding.window(Y,nprev,1,1)
table((ll1c$pred > .5), tail(yy,nprev))
ll.impt.plot(ll1c)
roc(ll1c,1)

set.seed(12345)
ll3c=lassologit.expanding.window(Y,nprev,1,3)
table((ll3c$pred > .5), tail(yy,nprev))
ll.impt.plot(ll3c)
roc(ll3c,1)

set.seed(12345)
ll6c=lassologit.expanding.window(Y,nprev,1,6)
table((ll6c$pred > .5), tail(yy,nprev))
ll.impt.plot(ll6c)
roc(ll6c,1)

set.seed(12345)
ll12c=lassologit.expanding.window(Y,nprev,1,12)
table((ll12c$pred > .5), tail(yy,nprev))
ll.impt.plot(ll12c)
roc(ll12c,1)


#######################
# Unconditional Mean  #
#######################
set.seed(12345)
um1c=um.expanding.window(Y,nprev,1,1)
sum(abs((um1c$pred > .5)-tail(yy,nprev)))
table((um1c$pred > .5), tail(yy,nprev))

um3c=um.expanding.window(Y,nprev,1,3)
sum(abs((um3c$pred > .5)-tail(yy,nprev)))
table((um3c$pred > .5), tail(yy,nprev))

um6c=um.expanding.window(Y,nprev,1,6)
sum(abs((um6c$pred > .5)-tail(yy,nprev)))
table((um6c$pred > .5), tail(yy,nprev))

um12c=um.expanding.window(Y,nprev,1,12)
sum(abs((um12c$pred > .5)-tail(yy,nprev)))
table((um12c$pred > .5), tail(yy,nprev))


#######################
#    Probit model     #
#######################
## Only with linear variables
Y=as.matrix(impt.data[,-1])
yy=Y[,1]
nprev=nrow(impt.data) - which(impt.data$yyyymm==200501) + 1
oosy=tail(yy,nprev)

set.seed(12345)
pm1c=pm.expanding.window(Y,nprev,1,1)
table((pm1c$pred > .5), tail(yy,nprev))

Sset.seed(12345)
pm3c=pm.expanding.window(Y,nprev,1,3)
table((pm3c$pred > .5), tail(yy,nprev))

set.seed(12345)
pm6c=pm.expanding.window(Y,nprev,1,6)
table((pm6c$pred > .5), tail(yy,nprev))

set.seed(12345)
pm12c=pm.expanding.window(Y,nprev,1,12)
table((pm12c$pred > .5), tail(yy,nprev))


#######################
#  E12 Visualization  #
#######################
data = read_xlsx("PredictorData2019.xlsx", sheet="Monthly")

## Creating date object
data$yyyymm <- as.Date(paste0(data$yyyymm, "01"), format ="%Y%m%d")

## Visualization
data <- data[which(data$yyyymm == "1974-12-01"):which(data$yyyymm == "2015-12-01"),]

scale <- mean(data$Index)/mean(data$E12)

## Finding Maxima points
last <- data[which(data$yyyymm == "2010-10-01"):which(data$yyyymm == "2015-12-01"),]
lastdate <- last[which.max(last$E12),]$yyyymm

mid <- data[which(data$yyyymm == "2005-10-01"):which(data$yyyymm == "2010-12-01"),]
middate <- mid[which.max(mid$E12),]$yyyymm

first <- data[which(data$yyyymm == "2000-10-01"):which(data$yyyymm == "2003-12-01"),]
firstdate <- first[which.max(first$E12),]$yyyymm

ggplot(data=data)+ 
  geom_line(aes(x=yyyymm, y= Index, color = "Index")) +
  geom_line(aes(x=yyyymm, y=E12*scale, color = "E12")) +
  geom_vline(xintercept = as.Date(lastdate), 
             linetype = "dashed", colour = "maroon") +
  geom_vline(xintercept = as.Date(middate), 
             linetype = "dashed", colour = "maroon") +
  geom_vline(xintercept = as.Date(firstdate), 
             linetype = "dashed", colour = "maroon") +
  theme(legend.title = element_blank()) +
  xlab("year") +
  theme(axis.title.y=element_blank()) +
  ggtitle("S&P Index & E12")

