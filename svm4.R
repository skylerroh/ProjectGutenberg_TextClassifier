library(e1071)
setwd("/accounts/grad/haolyu")
word_feature <- read.csv("doc_feature_matrix3_var.csv", header = T, stringsAsFactors = T)
word_feature <- word_feature[,-1]
row.names(word_feature)<- word_feature$X
model_svm <- svm(as.factor(tags)~., data = word_feature, kernel="radial",cost = 10,scale=F,cross = 5)
save(model_svm,file = "Varcost10.RData")
cost_list <- c(5,100,1000)
model_list <- lapply(cost_list,function(x){svm(as.factor(tags)~., data = word_feature, kernel="radial",cost = x,scale=F,cross = 5)})
save.image(file = "Varsvm4.RData")
pred <-  predict(model_svm,test)
y <- example_label_kaggle$category
a <- table(y,pred)
sum(diag(a))/sum(a)
