
# imporet packages
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import pylab
import scipy.stats as st
import statsmodels.formula.api as smf

# reading a csv file using pandas library
salarydata=pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\simple linear\\Salary_data.csv")
salarydata.columns
# print skewness,kurtosis and plot histogram, distplot, boxplot
# DeliveryTime
salarydata['YearsExperience'].skew()
salarydata['YearsExperience'].kurt()
plt.hist(salarydata['YearsExperience'],edgecolor='k')
sns.distplot(salarydata['YearsExperience'],hist=False)
plt.boxplot(salarydata['YearsExperience'])
# SortingTime
salarydata['Salary'].skew()
salarydata['Salary'].kurt()
plt.hist(salarydata['Salary'],edgecolor='k')
sns.distplot(salarydata['Salary'],hist=False)
plt.boxplot(salarydata['Salary'])
# plot scatter diagram
plt.plot(salarydata.YearsExperience,salarydata.Salary,"bo");plt.xlabel("YearsExperience");plt.ylabel("Salary")
# findout correlation value
salarydata.YearsExperience.corr(salarydata.Salary) 
np.corrcoef(salarydata.YearsExperience,salarydata.Salary)

# For preparing linear regression model 
# model 1
model=smf.ols("YearsExperience~Salary",data=salarydata).fit()
# For getting coefficients of the varibles used in equation
model.params
# P-values for the variables and R-squared value for prepared model
model.summary()
# 95% confidence interval
model.conf_int(0.05)
# Predicted values  the model
pred = model.predict(pd.DataFrame(salarydata.iloc[:,1]))
# Visualization of regresion line over the scatter plot of Waist and AT
# For visualization we need to import matplotlib.pyplot
plt.scatter(x=salarydata['Salary'],y=salarydata['YearsExperience'],color='red');plt.plot(salarydata['Salary'],pred,color='black');plt.xlabel('Salary');plt.ylabel('YearsExperience')
pred.corr(salarydata.YearsExperience)
# model 2
# Transforming variables for accuracy
model2 = smf.ols('YearsExperience~np.log(Salary)',data=salarydata).fit()
model2.params
model2.summary()
# 99% confidence level
print(model2.conf_int(0.01))
pred2 = model2.predict(pd.DataFrame(salarydata['Salary']))
pred2.corr(salarydata.YearsExperience)
# pred2 = model2.predict(deliverytime.iloc[:,1])
pred2
plt.scatter(x=salarydata['Salary'],y=salarydata['YearsExperience'],color='green');plt.plot(salarydata['Salary'],pred2,color='blue');plt.xlabel('Salary');plt.ylabel('YearsExperience')
# model 3
# Exponential transformation
model3 = smf.ols('np.log(YearsExperience)~Salary',data=salarydata).fit()
model3.params
model3.summary()
 # 99% confidence level
print(model3.conf_int(0.01))
pred_log = model3.predict(pd.DataFrame(salarydata['Salary']))
pred_log
# as we have used log(AT) in preparing model
pred3=np.exp(pred_log)
pred3
pred3.corr(salarydata.YearsExperience)
plt.scatter(x=salarydata['Salary'],y=salarydata['YearsExperience'],color='green');plt.plot(salarydata.Salary,np.exp(pred_log),color='blue');plt.xlabel('Salary');plt.ylabel('YearsExperience')
resid_3 = pred3-salarydata.YearsExperience
# we will consider the model having highest R-Squared value which is the log transformation
# getting residuals of the entire data set
experience_resid = model3.resid_pearson 
experience_resid
plt.plot(model3.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")
# Predicted vs actual values
plt.scatter(x=pred3,y=salarydata.YearsExperience);plt.xlabel("Predicted");plt.ylabel("Actual")

















































