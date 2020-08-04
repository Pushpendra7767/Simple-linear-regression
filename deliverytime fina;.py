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
deliverytime=pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\simple linear\\deliverytime.csv")
deliverytime.columns
# print skewness,kurtosis and plot histogram, distplot, boxplot
# DeliveryTime
deliverytime['DeliveryTime'].skew()
deliverytime['DeliveryTime'].kurt()
plt.hist(deliverytime['DeliveryTime'],edgecolor='k')
sns.distplot(deliverytime['DeliveryTime'],hist=False)
plt.boxplot(deliverytime['DeliveryTime'])
# SortingTime
deliverytime['SortingTime'].skew()
deliverytime['SortingTime'].kurt()
plt.hist(deliverytime['SortingTime'],edgecolor='k')
sns.distplot(deliverytime['SortingTime'],hist=False)
plt.boxplot(deliverytime['SortingTime'])
# plot scatter diagram
plt.plot(deliverytime.DeliveryTime,deliverytime.SortingTime,"bo");plt.xlabel("DeliveryTime");plt.ylabel("SortingTime")
# findout correlation value
deliverytime.DeliveryTime.corr(deliverytime.SortingTime) 
np.corrcoef(deliverytime.DeliveryTime,deliverytime.SortingTime)

# For preparing linear regression model 
# model 1
model=smf.ols("DeliveryTime~SortingTime",data=deliverytime).fit()
# For getting coefficients of the varibles used in equation
model.params
# P-values for the variables and R-squared value for prepared model
model.summary()
# 95% confidence interval
model.conf_int(0.05)
# Predicted values  the model
pred = model.predict(pd.DataFrame(deliverytime.iloc[:,1]))
# Visualization of regresion line over the scatter plot of Waist and AT
# For visualization we need to import matplotlib.pyplot
plt.scatter(x=deliverytime['SortingTime'],y=deliverytime['DeliveryTime'],color='red');plt.plot(deliverytime['SortingTime'],pred,color='black');plt.xlabel('SortingTime');plt.ylabel('DeliveryTime')
pred.corr(deliverytime.DeliveryTime)
# model 2
# Transforming variables for accuracy
model2 = smf.ols('DeliveryTime~np.log(SortingTime)',data=deliverytime).fit()
model2.params
model2.summary()
# 99% confidence level
print(model2.conf_int(0.01))
pred2 = model2.predict(pd.DataFrame(deliverytime['SortingTime']))
pred2.corr(deliverytime.DeliveryTime)
# pred2 = model2.predict(deliverytime.iloc[:,1])
pred2
plt.scatter(x=deliverytime['SortingTime'],y=deliverytime['DeliveryTime'],color='green');plt.plot(deliverytime['SortingTime'],pred2,color='blue');plt.xlabel('SortingTime');plt.ylabel('DeliveryTime')
# model 3
# Exponential transformation
model3 = smf.ols('np.log(DeliveryTime)~SortingTime',data=deliverytime).fit()
model3.params
model3.summary()
 # 99% confidence level
print(model3.conf_int(0.01))
pred_log = model3.predict(pd.DataFrame(deliverytime['SortingTime']))
pred_log
 # as we have used log(AT) in preparing model
pred3=np.exp(pred_log)
pred3
pred3.corr(deliverytime.DeliveryTime)
plt.scatter(x=deliverytime['SortingTime'],y=deliverytime['DeliveryTime'],color='green');plt.plot(deliverytime.SortingTime,np.exp(pred_log),color='blue');plt.xlabel('SortingTime');plt.ylabel('DeliveryTime')
resid_3 = pred3-deliverytime.DeliveryTime
# we will consider the model having highest R-Squared value which is the log transformation
# getting residuals of the entire data set
delivery_resid = model3.resid_pearson 
delivery_resid
plt.plot(model3.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")
# Predicted vs actual values
plt.scatter(x=pred3,y=deliverytime.DeliveryTime);plt.xlabel("Predicted");plt.ylabel("Actual")































