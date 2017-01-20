#download the data files to work directory before start
training <- read.csv(paste(getwd(), "/train.csv"))
testing <- read.csv(paste(getwd(), "/test.csv"))
submission_samp <- read.csv(paste(getwd(), "/sample_submission.csv"))
library(caret)
library(parallel)
library(doParallel)
library(randomForest)
library(gbm)
library(plyr)
library(data.table)

#combine training and testing so that we can remove all NA columns later on
alldata <-rbind(training, as.data.table(testing)[, SalePrice:=0])
nacolumns <- colSums(is.na(alldata))

good_index <- colnames(training[nacolumns == 0]) #keep columns without 'NA'
good_index <- good_index[-1] # id is irrelevant to the modelling
myDataNZV <- nearZeroVar(training[good_index], saveMetrics=TRUE) 
NZV_index <- row.names(subset(myDataNZV, nzv==T)) #find near zero variance variables
good_index <- good_index[!(good_index %in% NZV_index)] # remove near zero variance varibales

set.seed(1357)
inTrain <- createDataPartition(training$SalePrice, p = 0.75, list = FALSE)
mytraining <- training[inTrain, good_index]
mytraining$SalePrice <- log(mytraining$SalePrice + 1)
mytesting <- training[-inTrain, good_index]
mytesting$SalePrice <- log(mytesting$SalePrice + 1)

cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
fitControl <- trainControl(method = "cv", number = 10, allowParallel = TRUE)
set.seed(2468)
fit_gbm <- train(SalePrice ~ ., method="gbm", data = mytraining, trControl = fitControl,
                 verbose=F)
#set.seed(2468)
#fit_rf <- train(SalePrice ~ ., method="rf", data = mytraining, trControl = fitControl)
stopCluster(cluster)
registerDoSEQ()

#check the train RMSE
print(fit_gbm) 

#check the test RMSE
prediction <- predict(fit_gbm, mytesting)
errors <- sqrt(sum((prediction - mytesting$SalePrice)^2)/length(prediction))

#apply the model to the real test data
#the real test data are located at row 1461-2919 in alldata
prediction <- predict(fit_gbm, as.data.frame(alldata)[1461:2919, good_index])
submission_samp$SalePrice <- exp(prediction) - 1
write.csv(submission_samp, "housepricegbm.csv", row.names = F)


