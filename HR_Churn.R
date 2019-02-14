library(MASS)
library(tidyverse)
library(DataExplorer)
library(polycor)
library(caret)
library(recipes)
library(rsample)
library(ROCR)
library(car)
library(caret)
library(broom)

HW3Data <- HR_Churn

glimpse(HW3Data)

plot_missing(HW3Data)
dim(HW3Data)

HW3Data <- HW3Data %>%      
  select(Gone, everything())  

hetcor(HW3Data) #for correlation
glimpse(HW3Data)


#splitting the dataset into even sets

set.seed(2019)
train_test_split <- initial_split(HW3Data, prop = 0.5)
train_test_split

train_tbl <- training(train_test_split)
test_tbl <- testing(train_test_split)

dim(train_tbl)
dim(test_tbl)

#Retrieve train and test sets
rec_obj <- recipe(Gone ~ ., data = train_tbl) %>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>% 
  #step_bagimpute(all_predictors(), - all_outcomes()) %>%    #impute missing values
  prep(data = train_tbl)

#Print the recipe object
rec_obj

train_clean <- bake(rec_obj, new_data = train_tbl)
test_clean <- bake(rec_obj, new_data = test_tbl)

glimpse(train_clean)
plot_missing(train_clean)

#build logistic model
log.fit <- glm(Gone ~ ., data=train_clean, family = binomial(link="logit"), maxit = 100) # binomial is binary classification
summary(log.fit)

#build second logistic model relevant
log.fit2 <- glm(Gone ~ BusinessTravel_Non.Travel + BusinessTravel_Travel_Frequently + JobRole_Healthcare.Representative + BusinessTravel_Travel_Rarely + JobRole_Human.Resources + JobRole_Laboratory.Technician + JobRole_Manager + JobRole_Manufacturing.Director + JobRole_Research.Director + JobRole_Research.Scientist + JobRole_Sales.Executive + JobRole_Sales.Representative + MaritalStatus_Divorced + MaritalStatus_Married + MaritalStatus_Single + OverTime_No + OverTime_Yes + DistanceFromHome + EnvironmentSatisfaction + JobInvolvement + JobSatisfaction + NumCompaniesWorked + TotalWorkingYears + TrainingTimesLastYear + WorkLifeBalance + YearsInCurrentRole + YearsSinceLastPromotion, data=train_clean, family = binomial(link="logit"), maxit = 100) # binomial is binary classification
summary(log.fit2)
names(log.fit2)

log.prob = predict(log.fit2,newdata=test_clean,type='response')
log.pred = ifelse(log.prob>0.60,"Yes","No")

table(log.pred,test_clean$Gone)
mean(log.pred==test_clean$Gone)

pr = prediction(log.prob, test_clean$Gone)
prf = performance(pr,measure="tnr",x.measure="fnr")
plot(prf)

auc = performance(pr, measure="auc")
auc = auc@y.values[[1]]
auc

plot(log.fit)

durbinWatsonTest(log.fit2)
anova(log.fit2, test="Chisq") #helps generate a kia-squared or something

outlierTest(log.fit)
vif(log.fit)

#LDA Model
lda.fit = lda(Gone ~ BusinessTravel_Non.Travel + BusinessTravel_Travel_Frequently + JobRole_Healthcare.Representative + BusinessTravel_Travel_Rarely + JobRole_Human.Resources + JobRole_Laboratory.Technician + JobRole_Manager + JobRole_Manufacturing.Director + JobRole_Research.Director + JobRole_Research.Scientist + JobRole_Sales.Executive + JobRole_Sales.Representative + MaritalStatus_Divorced + MaritalStatus_Married + MaritalStatus_Single + OverTime_No + OverTime_Yes + DistanceFromHome + EnvironmentSatisfaction + JobInvolvement + JobSatisfaction + NumCompaniesWorked + TotalWorkingYears + TrainingTimesLastYear + WorkLifeBalance + YearsInCurrentRole + YearsSinceLastPromotion, data=train_clean)
summary(lda.fit)
names(lda.fit)

plot(lda.fit)

lda.pred = predict(lda.fit, test_clean)
names(lda.pred)

lda.class = lda.pred$class

table(lda.class,test_clean$Gone)
mean(lda.class==test_clean$Gone)

not#QDA model
qda.fit = qda(Gone ~ BusinessTravel_Non.Travel + BusinessTravel_Travel_Frequently + JobRole_Healthcare.Representative + BusinessTravel_Travel_Rarely + JobRole_Human.Resources + JobRole_Laboratory.Technician + JobRole_Manager + JobRole_Manufacturing.Director + JobRole_Research.Director + JobRole_Research.Scientist + JobRole_Sales.Executive + JobRole_Sales.Representative + MaritalStatus_Divorced + MaritalStatus_Married + MaritalStatus_Single + OverTime_No + OverTime_Yes + DistanceFromHome + EnvironmentSatisfaction + JobInvolvement + JobSatisfaction + NumCompaniesWorked + TotalWorkingYears + TrainingTimesLastYear + WorkLifeBalance + YearsInCurrentRole + YearsSinceLastPromotion, data=train_clean)
summary(qda.fit)
names(qda.fit)

plot(qda.fit)

qda.pred = predict(qda.fit, test_clean)
names(qda.pred)

qda.class = qda.pred$class

table(qda.class,test_clean$Gone)
mean(qda.class==test_clean$Gone)




