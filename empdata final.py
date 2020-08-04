
# import package
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import pylab
import scipy.stats as st

# reading a csv file using pandas library
empdata=pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\simple linear\\empdata.csv")
empdata.columns
# print skewness,kurtosis and plot histogram, distplot, boxplot
# Salaryhike
empdata['Salaryhike'].skew()
empdata['Salaryhike'].kurt()
plt.hist(empdata['Salaryhike'],edgecolor='k')
sns.distplot(empdata['Salaryhike'],hist=False)
plt.boxplot(empdata['Salaryhike'])
# Churnoutrate
empdata['Churnoutrate'].skew()
empdata['Churnoutrate'].kurt()
plt.hist(empdata['Churnoutrate'],edgecolor='k')
sns.distplot(empdata['Churnoutrate'],hist=False)
plt.boxplot(empdata['Churnoutrate'])
# plot scatter diagram
plt.plot(empdata.Salaryhike,empdata.Churnoutrate,"bo");plt.xlabel("Salaryhike");plt.ylabel("Churnoutrate")
# findout correlation value
empdata.Salaryhike.corr(empdata.Churnoutrate) 
np.corrcoef(empdata.Salaryhike,empdata.Churnoutrate)

# prepared simple linear regression model for r^2 value
# prepare model 1
model1 = LinearRegression()
model1.fit(empdata.Salaryhike.values.reshape(-1,1),empdata.Churnoutrate)
pred1 = model1.predict(empdata.Salaryhike.values.reshape(-1,1))
# Adjusted R-Squared value
model1.score(empdata.Salaryhike.values.reshape(-1,1),empdata.Churnoutrate)
rmse1 = np.sqrt(np.mean((pred1-empdata.Churnoutrate)**2)) 
model1.coef_
model1.intercept_
# visulization for distribution
plt.scatter(pred1,(pred1-empdata.Churnoutrate),c="r")
plt.hist(pred1-empdata.Churnoutrate,color='red',edgecolor='k')
st.probplot(pred1-empdata.Churnoutrate,dist="norm",plot=pylab)

# prepare model 2
# Fitting Quadratic Regression 
empdata["Salaryhike_sqrd"] = empdata.Salaryhike*empdata.Salaryhike
model2 = LinearRegression()
model2.fit(X = empdata.iloc[:,[0,2]],y=empdata.Churnoutrate)
pred2 = model2.predict(empdata.iloc[:,[0,2]])
# Adjusted R-Squared value
model2.score(empdata.iloc[:,[0,2]],empdata.Churnoutrate)
rmse2 = np.sqrt(np.mean((pred2-empdata.Churnoutrate)**2)) 
model2.coef_
model2.intercept_
# visulization for distribution
plt.scatter(pred2,(pred2-empdata.Churnoutrate),c="orange")
plt.hist(pred2-empdata.Churnoutrate,color='orange',edgecolor='k')
st.probplot(pred2-empdata.Churnoutrate,dist="norm",plot=pylab)

# prepare model 3
# Let us prepare a model by applying transformation on dependent variable
empdata["Churnoutrate_sqrt"] = np.sqrt(empdata.Churnoutrate)
model3 = LinearRegression()
model3.fit(X = empdata.iloc[:,[0,2]],y=empdata.Churnoutrate_sqrt)
pred3 = model3.predict(empdata.iloc[:,[0,2]])
# Adjusted R-Squared value
model3.score(empdata.iloc[:,[0,2]],empdata.Churnoutrate_sqrt)
rmse3 = np.sqrt(np.mean(((pred3)**2-empdata.Churnoutrate)**2))
model3.coef_
model3.intercept_
# visulization for distribution
plt.scatter((pred3)**2,((pred3)**2-empdata.Churnoutrate),c="green")
plt.hist((pred3)**2-empdata.Churnoutrate,color='green',edgecolor='k')
st.probplot((pred3)**2-empdata.Churnoutrate,dist="norm",plot=pylab)

# prepare model 4
# Let us prepare a model by applying transformation on dependent variable without transformation on input variables 
model4 = LinearRegression()
model4.fit(X = empdata.Salaryhike.values.reshape(-1,1),y=empdata.Churnoutrate_sqrt)
pred4 = model4.predict(empdata.Salaryhike.values.reshape(-1,1))
# Adjusted R-Squared value
model4.score(empdata.Salaryhike.values.reshape(-1,1),empdata.Churnoutrate_sqrt)
rmse4 = np.sqrt(np.mean(((pred4)**2-empdata.Churnoutrate)**2))
model4.coef_
model4.intercept_
# visulization for distribution
plt.scatter((pred4)**2,((pred4)**2-empdata.Churnoutrate),c="blue")
plt.hist((pred4)**2-empdata.Churnoutrate,edgecolor='k')
st.probplot((pred4)**2-empdata.Churnoutrate,dist="norm",plot=pylab)

