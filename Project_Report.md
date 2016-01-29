---
Title: Predictive Analysis on Self Movement 
  
Author: "by Sri Hasri"
output:
  html_document:
    fig_height: 9
    fig_width: 9
  pdf_document: default
  word_document: default
---
## GitHub Link

The repository of this assignment in Github located here:
https://github.com/smhasri/Module8_Project.

Publish online : http://rpubs.com/smhasri/147871



## Project Introduction  
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.  

In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise.  

## Data Preprocessing  

Install the following packages :

```{r, cache = T}
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)
```
### Download the Data

Retrieve the data :
```{r, cache = T}
trainUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainFile <- "./data/pml-training.csv"
testFile  <- "./data/pml-testing.csv"
if (!file.exists("./data")) {
  dir.create("./data")
}
if (!file.exists(trainFile)) {
  download.file(trainUrl, destfile=trainFile, method="curl")
}
if (!file.exists(testFile)) {
  download.file(testUrl, destfile=testFile, method="curl")
}
```  
### Read the Data
After downloading the data from the data source, we can read the two csv files into two data frames.  
```{r, cache = T}
trainRaw <- read.csv("./data/pml-training.csv")
testRaw <- read.csv("./data/pml-testing.csv")
dim(trainRaw)
dim(testRaw)
```
The training data set contains 19622 observations and 160 variables, while the testing data set contains 20 observations and 160 variables. The "classe" variable in the training set is the outcome to predict. 

### Clean the data
i) Clean the data and get rid of observations with missing values as well as some meaningless variables.
```{r, cache = T}
sum(complete.cases(trainRaw))
```
ii) remove columns that contain NA missing values.
```{r, cache = T}
trainRaw <- trainRaw[, colSums(is.na(trainRaw)) == 0] 
testRaw <- testRaw[, colSums(is.na(testRaw)) == 0] 
```  
iii) Remove columns that do not essential to the accelerometer measurements.
```{r, cache = T}
classe <- trainRaw$classe
trainRemove <- grepl("^X|timestamp|window", names(trainRaw))
trainRaw <- trainRaw[, !trainRemove]
trainCleaned <- trainRaw[, sapply(trainRaw, is.numeric)]
trainCleaned$classe <- classe
testRemove <- grepl("^X|timestamp|window", names(testRaw))
testRaw <- testRaw[, !testRemove]
testCleaned <- testRaw[, sapply(testRaw, is.numeric)]
```
Now, the cleaned training data set contains 19622 observations and 53 variables, while the testing data set contains 20 observations and 53 variables. The "classe" variable is still in the cleaned training set.

### Slice the data

Split the cleaned training set into a pure training data set (70%) and a validation data set (30%).  
```{r, cache = T}
set.seed(22519) # For reproducibile purpose
inTrain <- createDataPartition(trainCleaned$classe, p=0.70, list=F)
trainData <- trainCleaned[inTrain, ]
testData <- trainCleaned[-inTrain, ]
```

## Data Modeling

i) Predictive model for activity recognition using **Random Forest** algorithm because it automatically selects important variables and is robust to correlated covariates & outliers in general. 
ii) Use **5-fold cross validation** when applying the algorithm.  
```{r, cache = T}
controlRf <- trainControl(method="cv", 5)
modelRf <- train(classe ~ ., data=trainData, method="rf", trControl=controlRf, ntree=250)
modelRf
```
iii) Estimate the performance of the model on the validation data set.  
```{r, cache = T}
predictRf <- predict(modelRf, testData)
confusionMatrix(testData$classe, predictRf)
```
```{r, cache = T}
accuracy <- postResample(predictRf, testData$classe)
accuracy
oose <- 1 - as.numeric(confusionMatrix(testData$classe, predictRf)$overall[1])
oose
```
So, the estimated accuracy of the model is 99.42% and the estimated out-of-sample error is 0.58%.

## Predicting for Test Data Set
Apply the model to the original testing data set downloaded from the data source. Remove the `problem_id` column first.  
```{r, cache = T}
result <- predict(modelRf, testCleaned[, -length(names(testCleaned))])
result
```  

## Appendix: Figures
1. Correlation Matrix Visualization  
```{r, cache = T}
corrPlot <- cor(trainData[, -length(names(trainData))])
corrplot(corrPlot, method="color")
```
2. Decision Tree Visualization
```{r, cache = T}
treeModel <- rpart(classe ~ ., data=trainData, method="class")
prp(treeModel) # fast plot
```