# stat154 project Team 2
# November 2015
# code for Random Forest & SVM on Classifying Books

# open issues:

##############################
# to test, run the following
##############################
install.packages("randomForest")
install.packages("e1071")
library(randomForest)
library(MASS)
library(e1071)

#------------------------------------RF part:-----------------------------------------
# set up the training data
word_feature <- read.csv("doc_feature_matrix.csv", header = T, stringsAsFactors = T)
word_feature <- word_feature[,-1]
row.names(word_feature)<- word_feature$X
y <- factor(word_feature$tags)
df <- word_feature[, names(word_feature)!="tags"]
set.seed(1)
ntest <- floor(nrow(word_feature)*0.3)
index <- sample(1:nrow(word_feature),ntest) 
test <- df[index,]
ytest <- y[index]
train <- df[-index,]
ytrain <- y[-index]
#choose # of trees
ntree_list <- seq(10,300,by=50)
error_rate <- numeric(0)
for(n in ntree_list){
  model <- randomForest(ytrain~., data = train, mtry =45, ntree = n)
  yhat <- predict(model,test,type="response")
  confusion <- table(yhat,ytest)
  error_rate1 <- 1-sum(diag(confusion))/sum(confusion)
  error_rate <- c(error_rate,error_rate1)  
}
plot(ntree_list,error_rate,type = "l")
num <- which.min(error_rate)

# according to the plot we can see that the minimal tree can be 110
# cv Random Forest
cvRF <- function(data,nfold,m,d){
  set.seed(1)
  random <- sample(1:nrow(data))
  group <- split(random, cut(1:nrow(data), nfold))
  err <- sapply(group, function(x){
    train <- data[-x,]
    test <- data[x,]
    ytrain <- y[-x]
    ytest <- y[x]
    fit <- randomForest(ytrain~.,data = train, mtry = m,depth = d,ntree = num)
    yhat <- predict(fit,test,type="response")
    confusion <- table(yhat,ytest)
    1-sum(diag(confusion))/sum(confusion)
  })
  mean(err)
}
mtry_list <- seq(10,100,by=5)
depth_list <- seq(50,250,by=50)
error_list <- matrix(NA,ncol = length(depth_list),nrow = length(mtry_list))
for(i in 1:length(depth_list)){
  error_list[,i] <- sapply(depth_list,function(x){cvRF(df,10,x,depth_list[i])})
}
save(error_list,file = "error.Rda")

#------------------------------------SVM part:----------------------------------------
# cv SVM on gamma and cost
tune.out <- tune(svm, as.factor(tags)~., data = word_feature, kernel="radial",
                 ranges = list(cost = c(0.1,1,5,10,100,1000),gamma = c(1e-5,5e-5,1e-4,5e-4,1e-3,1e-2)), scale = F)
summary(tune.out)