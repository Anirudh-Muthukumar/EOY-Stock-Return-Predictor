#The dataset contains only monthly stock returns from second largest stock exchange in the world, NASDAQ stock exchange. The years included are 2000-2009. The ultimate goal is to predict whether or not the stock return in December will be positive or not, based on analysis of stock returns for the first 11 months of the year.

stocks = read.csv("StocksCluster.csv")
str(stocks)
summary(stocks)
#Total of 11580 observations in the dataset which consist of 12 variables. Among these, 11 are independent variables which denotes the monthly returns of the company and the dependent variable is a binary variable which takes value 1 if return was positive, and 0 if negative.

#Proportion of companies having positive return in Decemeber
t = table(stocks$PositiveDec)
t[2]/sum(t)

#Exploring the correlation between any two return variables
cor(stocks[-12])
sort(abs(cor(stocks)))
#maximum correlation between any two return variable is 0.19

#Month having largest and smallest mean return
sort(colMeans(stocks))
#April has largest mean return while September has lowest mean return

#Initial logistic regression model
library(caTools)
set.seed(144)
#Splitting the data into training and testing set with ratio 70:30
split = sample.split(stocks$PositiveDec, SplitRatio = 0.7)

stocksTrain = subset(stocks, split==T)
stocksTest = subset(stocks, split==F)

#Using all the variables to predict the outcome variable
stocksLR = glm( PositiveDec ~ ., data = stocksTrain, family = binomial)
summary(stocksLR)
#Seems quite a many variables are significant in predicting the outcome variable
predLR = predict(stocksLR, newdata = stocksTest, type="response")
t = table(stocksTest$PositiveDec, predLR>0.5)
t
#Accuracy of the model on test data using threshold of 0.5
sum(diag(t))/sum(t)

#Accuracy on the test set of baseline model
t = table(stocks$PositiveDec)
t[2]/sum(t)
#Our initially model has very slightly improved on the baseline model

#Clustering the stocks to improve our model

#First step in clustering is to remove the dependent variable 
limitedTrain = stocksTrain
limitedTrain$PositiveDec = NULL
limitedTest = stocksTest
limitedTest$PositiveDec = NULL

#Normalizing the data before clustering
library(caret)
preprocess = preProcess(limitedTrain)
normTrain = predict(preprocess, limitedTrain)
normTest = predict(preprocess, limitedTest)

summary(normTrain)
summary(normTest)
#Since the distribution of return variables are different in normalized train set and test set, we have mean values of return variables much closer to 0 in train than in test.

set.seed(144)
k = 3
stocksKM = kmeans(normTrain, centers = k)
table(stocksKM$cluster)
#cluster2 has largest number of data points while cluster3 has the lowest

#Using the trained model to predict the clusters of test set
library(flexclust)

km.kcca = as.kcca(stocksKM, normTrain)
kmTrain = predict(km.kcca)
kmTest = predict(km.kcca, newdata = normTest)
table(kmTest)
#2080 test-set observations were assigned to cluster 2

#Cluster-specific Predictions

stocksTrain1 = subset(stocksTrain, kmTrain==1)
stocksTrain2 = subset(stocksTrain, kmTrain==2)
stocksTrain3 = subset(stocksTrain, kmTrain==3)

stocksTest1 = subset(stocksTest, kmTest==1)
stocksTest2 = subset(stocksTest, kmTest==2)
stocksTest3 = subset(stocksTest, kmTest==3)

mean(stocksTrain1$PositiveDec)
mean(stocksTrain2$PositiveDec)
mean(stocksTrain3$PositiveDec)
#cluster1 has the highest average value of dependent variable in train-set

#Builing logistic regression models for each cluster
stockModel1 = glm( PositiveDec ~., data = stocksTrain1, family=binomial)
stockModel2 = glm( PositiveDec ~., data = stocksTrain2, family=binomial)
stockModel3 = glm( PositiveDec ~., data = stocksTrain3, family=binomial)
summary(stockModel1)
summary(stockModel2)
summary(stockModel3)

predictTest1 = predict(stockModel1, newdata = stocksTest1, type="response")
predictTest2 = predict(stockModel2, newdata = stocksTest2, type="response")
predictTest3 = predict(stockModel3, newdata = stocksTest3, type="response")

t1 = table(stocksTest1$PositiveDec, predictTest1>0.5)
t1
t2 = table(stocksTest2$PositiveDec, predictTest2>0.5)
t2
t3 = table(stocksTest3$PositiveDec, predictTest3>0.5)
t3

#Accuracy of stockModel1 using threshold of 0.5
sum(diag(t1))/sum(t1)

#Accuracy of stockModel2 using threshold of 0.5
sum(diag(t2))/sum(t2)

#Accuracy of stockModel3 using threshold of 0.5
sum(diag(t3))/sum(t3)


#Computing the overall accuracy
AllPredictions = c(predictTest1, predictTest2, predictTest3)
AllOutcomes = c(stocksTest1$PositiveDec, stocksTest2$PositiveDec, stocksTest3$PositiveDec)

t = table(AllOutcomes, AllPredictions>0.5)
sum(diag(t))/sum(t)

#It can be inferred that this model has modest improvement over the initial logisitic regression model by cluster-specific prediction. Since predicting stock returns is a byzantine task which depends on a number of factors, this is a good increase in our accuracy by considering only the returns from previous calendar months. By investing only in the stocks about which we are more confident( higher predicted probablities), this cluster-specific prediction model has an edge for initial logistic regression model.
