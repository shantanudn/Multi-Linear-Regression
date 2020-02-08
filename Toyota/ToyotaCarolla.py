# Multilinear Regression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# loading the data
toyota = pd.read_csv("D:/Training/ExcelR_2/Multi_Linear_Regression/Toyota/ToyotaCorolla.csv")

# to get top 40 rows
toyota.head(40) 


##### EDA ###################3
toyota.columns

# Correlation matrix 
toyota.corr()

np.mean(toyota)
toyota['Price'].mean() 
toyota['Price'].median()
toyota['Price'].mode()
toyota['Price'].var()
toyota['Price'].std()

print(toyota.describe())
descriptive = toyota.describe()

from tabulate import tabulate as tb
print(tb(descriptive,toyota.columns))

######### boxplots ###########

plt.boxplot(toyota.Price)
plt.xticks([1,], ['Price'])
plt.boxplot(toyota.Age_08_04)
plt.xticks([1,], ['Age'])
plt.boxplot(toyota.KM)
plt.xticks([1,], ['KM'])
plt.boxplot(toyota.HP)
plt.xticks([1,], ['HP'])
plt.boxplot(toyota.cc)
plt.xticks([1,], ['cc'])
plt.boxplot(toyota.Doors)
plt.xticks([1,], ['Doors'])
plt.boxplot(toyota.Gears)
plt.xticks([1,], ['Gears'])
plt.boxplot(toyota.Quarterly_Tax)
plt.xticks([1,], ['Quarterly_Tax'])
plt.boxplot(toyota.Weight)
plt.xticks([1,], ['Weight'])

######### Histogram ###########
plt.hist(toyota.Price)
plt.xlabel('Price')
plt.hist(toyota.Age_08_04)
plt.xlabel('Age_08_04')
plt.hist(toyota.KM)
plt.xlabel('KM')
plt.hist(toyota.HP)
plt.xlabel('HP')
plt.hist(toyota.cc)
plt.xlabel('cc')
plt.hist(toyota.Doors)
plt.xlabel('Doors')
plt.hist(toyota.Gears)
plt.xlabel('Gears')
plt.hist(toyota.Quarterly_Tax)
plt.xlabel('Quarterly_Tax')
plt.hist(toyota.Weight)
plt.xlabel('Price')

#Scatter Plots
plt.plot(toyota.Age_08_04,toyota.Price,"ro");plt.xlabel("Age_08_04");plt.ylabel("Price")
plt.plot(toyota.KM,toyota.Price,"ro");plt.xlabel("KM");plt.ylabel("Price")
plt.plot(toyota.HP,toyota.Price,"ro");plt.xlabel("HP");plt.ylabel("Price")
plt.plot(toyota.cc,toyota.Price,"ro");plt.xlabel("cc");plt.ylabel("Price")
plt.plot(toyota.Doors,toyota.Price,"ro");plt.xlabel("Doors");plt.ylabel("Price")
plt.plot(toyota.Gears,toyota.Price,"ro");plt.xlabel("Gears");plt.ylabel("Price")
plt.plot(toyota.Quarterly_Tax,toyota.Price,"ro");plt.xlabel("Quarterly_Tax");plt.ylabel("Price")
plt.plot(toyota.Weight,toyota.Price,"ro");plt.xlabel("Weight");plt.ylabel("Price")


# Correlation matrix 
toyota.corr()

plt.matshow(toyota.corr())
plt.show()


# getting boxplot of price with respect to each category of gears 
import seaborn as sns 
sns.boxplot(x="Age_08_04",y="Price",data=toyota)

heat1 = toyota.corr()
sns.heatmap(heat1, xticklabels=toyota.columns, yticklabels=toyota.columns, annot=True)


# Scatter plot between the variables along with histograms
sns.pairplot(toyota)


# columns names
toyota.columns

# pd.tools.plotting.scatter_matrix(toyota); -> also used for plotting all in one graph
                             
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
# Preparing model                  
ml1 = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota).fit() # regression model

# Getting coefficients of variables               
ml1.params

# Summary
ml1.summary()

#The variables Doors and cc have p-values greater than 0.05, thus these variables insignificant to the outcome


# preparing model based only on Doors
ml_d=smf.ols('Price~Doors',data = toyota).fit()  
ml_d.summary() 


# Preparing model based only on cc
ml_cc=smf.ols('Price~cc',data = toyota).fit()  
ml_cc.summary() 

# Preparing model based only on Doors & cc
ml_dc=smf.ols('Price~Doors+cc',data = toyota).fit()  
ml_dc.summary() 

# Preparing model based only on Age_08_04,Doors & cc
ml_ac=smf.ols('Price~Age_08_04+Doors+cc',data = toyota).fit()  
ml_ac.summary() 

# Preparing model based only on KM,Doors & cc
ml_kdc=smf.ols('Price~KM+Doors+cc',data = toyota).fit()  
ml_kdc.summary() 

# Preparing model based only on HP,Doors & cc
ml_hdc=smf.ols('Price~HP+Doors+cc',data = toyota).fit()  
ml_hdc.summary() 

# Preparing model based only on Gears,Doors & cc
ml_gdc=smf.ols('Price~Gears+Doors+cc',data = toyota).fit()  
ml_gdc.summary() 

# Preparing model based only on Quarterly_Tax,Doors & cc
ml_qdc=smf.ols('Price~Quarterly_Tax+Doors+cc',data = toyota).fit()  
ml_qdc.summary() 

# Preparing model based only on Weight,Doors & cc
ml_wdc=smf.ols('Price~Weight+Doors+cc',data = toyota).fit()  
ml_wdc.summary() 

# Preparing model based only on Weight & Doors
ml_wd=smf.ols('Price~Weight+Doors',data = toyota).fit()  
ml_wd.summary() 

# Preparing model based only on Weight & cc
ml_wc=smf.ols('Price~Weight+cc',data = toyota).fit()  
ml_wc.summary() 

#Dropping variable "doors" since it is insignificant to the outcome
ml1_v2 = smf.ols('Price~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight',data=toyota).fit() # regression model
ml1_v2.summary()

#Dropping variable "cc" since it is insignificant to the outcome
ml1_v3 = smf.ols('Price~Age_08_04+KM+HP+Gears+Quarterly_Tax+Weight',data=toyota).fit() # regression model
ml1_v3.summary()
ml1_v3.params
#The model ml1_v3 produces a model of R-sqaured value of 0.864

# The above model provides an R-sqaured value of 0.864 and an Adj. R-squared value of 0.863
#Implementing the above model in the prediction model of regression

# Checking whether data has any influential values 
# influence index plots

import statsmodels.api as sm
sm.graphics.influence_plot(ml1)

# data in rows 601, 960, 221 & 80 are found to be influential to the dataframe, hence we drop these variable to improve the model accuracy

# Delete the rows with labels 601, 960, 221 & 80
toyota = toyota.drop([80,221,960,601], axis=0)

# final model
final_model = smf.ols('Price~Age_08_04+KM+HP+Gears+Quarterly_Tax+Weight',data=toyota).fit() 

sm.graphics.influence_plot(final_model)

final_model.params
final_model.summary() 
price_pred = final_model.predict(toyota)

final_model = smf.ols('Price~Age_08_04+KM+HP+Gears+Weight',data=toyota).fit() 


final_model.params
final_model.summary() 
price_pred = final_model.predict(toyota)

########    Normality plot for residuals ######
# histogram
plt.hist(final_model.resid_pearson) #
plt.xticks([1,], ['Residuals'])
#Transformation models 

######  Linearity #########
# Observed values VS Fitted values
plt.scatter(toyota.Price,price_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

#Residuals Vs Fitted Values
plt.scatter(price_pred,final_model.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


########    Normality plot for residuals ######
# histogram
plt.hist(final_model.resid_pearson) #

#Transformation models 

############################################## Log Transformation ######################

ml1_log = smf.ols('Price~np.log(Age_08_04)+np.log(KM)+np.log(HP)+np.log(cc)+np.log(Doors)+np.log(Gears)+np.log(Quarterly_Tax)+np.log(Weight)',data=toyota).fit() # regression model
ml1_log.summary()


######### Removing the variable "Gears" since it is insignifcant to the outcome #####

ml1_log = smf.ols('Price~np.log(Age_08_04)+np.log(KM)+np.log(HP)+np.log(cc)+np.log(Doors)+np.log(Quarterly_Tax)+np.log(Weight)',data=toyota).fit() # regression model
ml1_log.summary()

ml1_log.params
price_pred_log = ml1_log.predict(toyota)


###### Log Linearity #########
# Observed values VS Fitted values
plt.scatter(toyota.Price,price_pred_log,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

#Residuals Vs Fitted Values
plt.scatter(price_pred_log,ml1_log.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")



############################################## Exponential Model ####################


exp_model = smf.ols('np.log(Price)~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota).fit() 
exp_model.params
exp_model.summary() 

######### Removing the variable "Doors" since it is insignifcant to the outcome #####

exp_model = smf.ols('np.log(Price)~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight',data=toyota).fit() 
exp_model.params
exp_model.summary() 

print(exp_model.conf_int(0.05)) # 95% confidence level
pred_log = exp_model.predict(toyota)
pred_log
exp_pred=np.exp(pred_log)  # as we have used log(AT) in preparing model so we need to convert it back
exp_pred
exp_pred.corr(toyota.Price)
resid_3 = exp_pred-toyota.Price


######  Linearity for Exponential model #########
# Observed values VS Fitted values
plt.scatter(toyota.Price,exp_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

#Residuals Vs Fitted Values
plt.scatter(exp_pred,exp_model.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


########    Normality plot for residuals in Exponential model ######
# histogram
plt.hist(exp_model.resid_pearson) #






### Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
toyota_train,toyota_test  = train_test_split(toyota,test_size = 0.2) # 20% size

# preparing the model on train data 

model_train = smf.ols('np.log(Price)~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight',data=toyota).fit()

# train_data prediction
train_pred = model_train.predict(toyota_train)

# train residual values 
train_resid  = train_pred - toyota_train.Price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))

# prediction on test data set 
test_pred = model_train.predict(toyota_test)

# test residual values 
test_resid  = test_pred - toyota_test.Price

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
