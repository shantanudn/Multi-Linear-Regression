# Multilinear Regression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# loading the data
Computer_Data = pd.read_csv("D:/Training/ExcelR_2/Multi_Linear_Regression/Computer_Data/Computer_Data.csv", index_col=0)


# to get top 40 rows
Computer_Data.head(40) 


##### EDA ###################
Computer_Data.columns

Computer_Data.dtypes

# Correlation matrix 
Computer_Data.corr()

np.mean(Computer_Data)
Computer_Data['price'].mean() 
Computer_Data['price'].median()
Computer_Data['price'].mode()
Computer_Data['price'].var()
Computer_Data['price'].std()

print(Computer_Data.describe())
descriptive = Computer_Data.describe()

from tabulate import tabulate as tb
print(tb(descriptive,Computer_Data.columns))

######### boxplots ###########

import seaborn as sns

plt.boxplot(Computer_Data.price)
plt.xticks([1,], ['price'])
plt.boxplot(Computer_Data.speed)
plt.xticks([1,], ['speed'])
plt.boxplot(Computer_Data.hd)
plt.xticks([1,], ['hd'])
plt.boxplot(Computer_Data.ram)
plt.xticks([1,], ['ram'])
plt.boxplot(Computer_Data.screen)
plt.xticks([1,], ['screen'])
plt.boxplot(Computer_Data.ads)
plt.xticks([1,], ['ads'])
plt.boxplot(Computer_Data.trend)
plt.xticks([1,], ['trend'])

######### Histogram ###########
plt.hist(Computer_Data.price)
plt.xlabel('price')
plt.hist(Computer_Data.speed)
plt.xlabel('speed')
plt.hist(Computer_Data.hd)
plt.xlabel('hd')
plt.hist(Computer_Data.ram)
plt.xlabel('ram')
plt.hist(Computer_Data.screen)
plt.xlabel('screen')
plt.hist(Computer_Data.ads)
plt.xlabel('ads')
plt.hist(Computer_Data.trend)
plt.xlabel('trend')

#Scatter Plots

plt.plot(Computer_Data.speed,Computer_Data.price,"ro");plt.xlabel("speed");plt.ylabel("price")
plt.plot(Computer_Data.hd,Computer_Data.price,"ro");plt.xlabel("hd");plt.ylabel("price")
plt.plot(Computer_Data.ram,Computer_Data.price,"ro");plt.xlabel("ram");plt.ylabel("price")
plt.plot(Computer_Data.screen,Computer_Data.price,"ro");plt.xlabel("screen");plt.ylabel("price")
plt.plot(Computer_Data.ads,Computer_Data.price,"ro");plt.xlabel("ads");plt.ylabel("price")
plt.plot(Computer_Data.trend,Computer_Data.price,"ro");plt.xlabel("trend");plt.ylabel("price")

# Correlation matrix 
Computer_Data.corr()

plt.matshow(Computer_Data.corr())
plt.show()


###### Creating dummy varibales for screen ########
Computer_Data_cd_dummies = pd.get_dummies(Computer_Data.cd)
Computer_Data_cd_dummies = Computer_Data_cd_dummies.rename(columns={"yes": "cd"})
Computer_Data_cd_dummies = Computer_Data_cd_dummies.drop(["no"],axis=1)

Computer_Data_multi_dummies = pd.get_dummies(Computer_Data.multi)
Computer_Data_multi_dummies = Computer_Data_multi_dummies.rename(columns={"yes": "multi"})
Computer_Data_multi_dummies = Computer_Data_multi_dummies.drop(["no"],axis=1)


Computer_Data_premium_dummies = pd.get_dummies(Computer_Data.premium)
Computer_Data_premium_dummies = Computer_Data_premium_dummies.rename(columns={"yes": "premium"})
Computer_Data_premium_dummies = Computer_Data_premium_dummies.drop(["no"],axis=1)


######## dropping the variables "cd","multi" and "multi" ##############
Computer_Data = Computer_Data.drop(["cd","multi","premium"],axis=1)

####### concatinating or combining the dummy variables column to the Computer_Data dataset
Computer_Data = pd.concat([Computer_Data,Computer_Data_cd_dummies,Computer_Data_multi_dummies,Computer_Data_premium_dummies], axis=1)


# getting boxplot of price with respect to each category of gears 
import seaborn as sns 
sns.boxplot(x="speed",y="price",data=Computer_Data)

heat1 = Computer_Data.corr()
sns.heatmap(heat1, xticklabels=Computer_Data.columns, yticklabels=Computer_Data.columns, annot=True)


# Scatter plot between the variables along with histograms
sns.pairplot(Computer_Data)


# columns names
Computer_Data.columns

# pd.tools.plotting.scatter_matrix(Computer_Data); -> also used for plotting all in one graph
                             
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model



### Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
Computer_Data_train,Computer_Data_test  = train_test_split(Computer_Data,test_size = 0.2) # 20% size

# preparing the model on train data 

model_train = smf.ols('price~speed+ram+hd+screen+ads+trend+cd+multi+premium',data=Computer_Data_train).fit() 
model_train.summary()


# train_data prediction
train_pred = model_train.predict(Computer_Data_train)

# train residual values 
train_resid  = train_pred - Computer_Data_train.price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))

# prediction on test data set 
test_pred = model_train.predict(Computer_Data_test)

# test residual values 
test_resid  = test_pred - Computer_Data_test.price

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))

print(tb(test_resid,test_rmse))

########    Normality plot for residuals ######
# histogram
plt.hist(model_train.resid_pearson) #
plt.xticks([1,], ['Residuals'])
#Transformation models 

######  Linearity #########
# Observed values VS Fitted values
plt.scatter(Computer_Data_test.price,test_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

#Residuals Vs Fitted Values
plt.scatter(test_pred,test_resid,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


###### Transforming the models #############
#############################################
############################################## Log Transformation ######################


# preparing the model on train data 

######## dropping the variables "cd","multi" and "premium" ##############
Computer_Data_log = Computer_Data_train.drop(["cd","multi","premium"],axis=1)
Computer_Data_log_test = Computer_Data_test.drop(["cd","multi","premium"],axis=1)

model_train_log = smf.ols('price~np.log(speed)+np.log(ram)+np.log(hd)+np.log(screen)+np.log(ads)+np.log(trend)',data=Computer_Data_log).fit() 
model_train_log.summary()


# train_data prediction
train_pred_log = model_train_log.predict(Computer_Data_log)

# train residual values 
train_resid_log  = train_pred_log - Computer_Data_log.price

# RMSE value for train data 
train_rmse_log = np.sqrt(np.mean(train_resid_log*train_resid_log))

# prediction on test data set 
test_pred_log = model_train_log.predict(Computer_Data_log_test)

# test residual values 
test_resid_log  = test_pred_log - Computer_Data_log_test.price

# RMSE value for test data 
test_rmse_log = np.sqrt(np.mean(test_resid_log*test_resid_log))



# histogram
plt.hist(model_train_log.resid_pearson) #

###### Log Linearity #########
# Observed values VS Fitted values
plt.scatter(Computer_Data_log_test.price,test_pred_log,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

#Residuals Vs Fitted Values
plt.scatter(test_pred_log,test_resid_log,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")



############################################## Exponential Model ####################


# preparing the model on train data 

######## dropping the variables "cd","multi" and "premium" ##############


model_train_exp = smf.ols('np.log(price)~speed+ram+hd+screen+ads+trend+cd+multi+premium',data=Computer_Data_train).fit()  
model_train_exp.summary()


# train_data prediction
train_pred_elog = model_train_exp.predict(Computer_Data_train)
train_pred_exp = np.exp(train_pred_elog) #Converting from log


# train residual values 
train_resid_exp  = train_pred_exp - Computer_Data_train.price

# RMSE value for train data 
train_rmse_exp = np.sqrt(np.mean(train_resid_exp*train_resid_exp))

# prediction on test data set 
test_pred_exp = model_train_exp.predict(Computer_Data_test)

# test residual values 
test_resid_exp  = test_pred_exp - Computer_Data_test.price

# RMSE value for test data 
test_rmse_exp = np.sqrt(np.mean(test_resid_exp*test_resid_exp))




########    Normality plot for residuals ######
# histogram
plt.hist(model_train_exp.resid_pearson) #
plt.xticks([1,], ['Residuals'])
#Transformation models 

######  Linearity #########
# Observed values VS Fitted values
plt.scatter(Computer_Data_test.price,test_pred_exp,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

#Residuals Vs Fitted Values
plt.scatter(test_pred_exp,test_resid_exp,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


#### RMSE Table ###########

# intialise data of lists. 
R_data = {'Train RMSE':[274.20], 'Test RMSE':[279.03], 'Train RMSE Exp': [272.55],'Test RMSE Exp':[227.06],'Train RMSE Log': [326.87],'Test RMSE log':[331.23]} 
  
# Create DataFrame 
RMSE_data = pd.DataFrame(R_data) 
  
# Print the output. 
print(RMSE_data)
print(tb(RMSE_data))
