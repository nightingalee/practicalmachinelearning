# Practical Machine Learning Course Project: quantifying exercise quality
Chunle Xiong  
17 January 2017  



## Overview

Using modern devices people can collect a large amount of data about personal exercise activities. Usually this can only quantify how much of a particular activity they do, but they rarely quantify how well they do it. The question that the machine learning prediction model in this course project will answer is how well people do activities according to their activity pattern.

Since the question has been identified, the next step is to choose the most relevant data to build the prediction model. We will use data from accelerometers on the belt, forearm, arm, and dumbell of six male young health participants aged between 20-28 years. They were asked to perform one set of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

By choosing the most relevant measurement data and building a proper prediction model, we can predict which class a participant's activities belong to and thus quantify how well he does the activities.

## Loading the data sets and relevant packages
After a quick exploratory analysis of the data sets by simply reading them using default read.csv settings, we have found a lot of missing values in the forms of "NA", "#DIV/0!" and "". So we load the data sets in the following way.


```r
#Loading the csv files that have been saved in the working directory
training <- read.csv(paste(getwd(), "/pml-training.csv", sep = ""), na.strings = c("NA", "#DIV/0!", ""))
quiz <- read.csv(paste(getwd(), "/pml-testing.csv", sep = ""), na.strings = c("NA", "#DIV/0!", ""))
#Loading the required packages for analysis
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(rpart)
library(rattle)
```

```
## Rattle: A free graphical interface for data mining with R.
## XXXX 4.1.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(parallel)
library(doParallel)
```

```
## Loading required package: foreach
```

```
## Loading required package: iterators
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

## Cleaning the data sets
A further exploratory analysis shows that some columns (i.e., variables) include a large portion of missing values (i.e., NA) and the first column only contains the row number which is irrelevant to the analysis. The first step of cleaning data is to remove these columns. The method used here removes the columns that contain at least one NA. This method works well for this particular project because the removed columns contain many NAs. Other methods need to be considered if only very few NAs are in the data sets.


```r
#Only getting the index, keeping the data sets unchanged
good_index <- colnames(training[colSums(is.na(training)) == 0])
good_index <- good_index[-1]
```

The second step of cleaning data is to remove NearZeroVariance variables using the following R code.


```r
myDataNZV <- nearZeroVar(training[good_index], saveMetrics=TRUE)
#myDataNZV shows column 5 corresponds to NearZeroVariance variables
good_index <- good_index[-5]
```

## Splitting the data set to training and testing sets
To make the analysis reproducible, we set a seed for data splitting and only keep the good data in the training and testing sets for analysis and prediction.


```r
set.seed(1357)
inTrain <- createDataPartition(training$classe, p = 0.6, list = FALSE)
#Only keeping the good data
mytraining <- training[inTrain, good_index]
mytesting <- training[-inTrain, good_index]
dim(training)
```

```
## [1] 19622   160
```

```r
dim(mytraining)
```

```
## [1] 11776    58
```

```r
dim(mytesting)
```

```
## [1] 7846   58
```

It can be seen from the displayed dimensions that the data sets that will be used for analysis have reduced number of variables from 160 to 58. The remaining 58 variables will be used as predictors for building up the model. The data cleaning does not only remove the irrelevant data and also makes the following analysis much more efficient.

## Classficiation tree analysis
Using the table function, we find that the outcome data is categorical. So we can use classfication trees or Random Forest to make the model. Before we try the more accurate while slower Random Forest method, we try the classfication tree rpart first to see if the accuracy is acceptable.


```r
table(mytraining$classe)
```

```
## 
##    A    B    C    D    E 
## 3348 2279 2054 1930 2165
```

```r
set.seed(12468)
rpart_fit <- train(classe ~ ., method="rpart", data = mytraining, 
                   trControl = trainControl(method = "cv", number = 4))

rpart_pred <- predict(rpart_fit, mytesting)
fancyRpartPlot(rpart_fit$finalModel,cex=.5)
```

![](index_files/figure-html/rpart analysis-1.png)<!-- -->

```r
confusionMatrix(rpart_pred, mytesting$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1707  386   35   51   14
##          B  139  607  144  206  355
##          C  250  246 1150  432  326
##          D  129  279   39  597   85
##          E    7    0    0    0  662
## 
## Overall Statistics
##                                          
##                Accuracy : 0.602          
##                  95% CI : (0.591, 0.6128)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.4974         
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7648  0.39987   0.8406  0.46423  0.45908
## Specificity            0.9134  0.86662   0.8064  0.91890  0.99891
## Pos Pred Value         0.7784  0.41833   0.4784  0.52879  0.98954
## Neg Pred Value         0.9071  0.85754   0.9599  0.89742  0.89132
## Prevalence             0.2845  0.19347   0.1744  0.16391  0.18379
## Detection Rate         0.2176  0.07736   0.1466  0.07609  0.08437
## Detection Prevalence   0.2795  0.18493   0.3064  0.14389  0.08527
## Balanced Accuracy      0.8391  0.63325   0.8235  0.69157  0.72900
```

According to the confusionMatrix, the accuracy of prediction on the out sample data is only 60.2%, which is too poor to be used for predicting 20 samples for the quiz.

## Use parallel implementation of Random Forest

Since the faster classification tree method does not provide enough accuracy, we try Random Forest to see if the accuracy can be improved. To make the modelling quicker, we choose the parallel implementation of Random Forest in the caret package with the resampling method changed from the default bootstrapping to 4-fold cross-validation.


```r
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv", number = 4, allowParallel = TRUE)
set.seed(2468)
fit <- train(classe ~ ., method="rf", data = mytraining, trControl = fitControl)
stopCluster(cluster)
registerDoSEQ()
```

## Out of sample error

Now we check the accuracy of the model. It can be seen that the accuracy is as high as 99.91% even applying the model to the testing sample set that is out of the training data set. This indicates that the out of sample error is as low as 0.09%. It should be noted that even though the testing data set is out of training data set, they belong to the same original data set. This means these data will have high degrees of consistency. If we apply the model to some data that is completely different from the original data set, for example the participants are different people, we might get less accurate prediction.


```r
prediction <- predict(fit, mytesting)
confusionMatrix(prediction, mytesting$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    0    0    0    0
##          B    0 1518    2    0    0
##          C    0    0 1366    4    0
##          D    0    0    0 1281    0
##          E    0    0    0    1 1442
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9991          
##                  95% CI : (0.9982, 0.9996)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9989          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   0.9985   0.9961   1.0000
## Specificity            1.0000   0.9997   0.9994   1.0000   0.9998
## Pos Pred Value         1.0000   0.9987   0.9971   1.0000   0.9993
## Neg Pred Value         1.0000   1.0000   0.9997   0.9992   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1935   0.1741   0.1633   0.1838
## Detection Prevalence   0.2845   0.1937   0.1746   0.1633   0.1839
## Balanced Accuracy      1.0000   0.9998   0.9990   0.9981   0.9999
```

## Apply the model to the quiz set


```r
myquiz <- quiz
colnames(myquiz)[160] <- "classe"
myquiz <- myquiz[, good_index]
myquiz_pred <- predict(fit, myquiz)
print(myquiz_pred)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

## Conclusion
The Random Forest model provides much more accurate prediction for the exercise quality data than the rpart model, and the accuracy is as high as 99.91%. It should be noted that for completely different data sets, the model might not be able to provide as accurate prediction as for the one from the same population that the training set is from.
