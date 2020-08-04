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
calories=pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\simple linear\\caloriesconsumed2.csv")
calories.columns
# print skewness,kurtosis and plot histogram, distplot, boxplot
# Weightgainedgrams
calories['Weightgainedgrams'].skew()
calories['Weightgainedgrams'].kurt()
plt.hist(calories['Weightgainedgrams'],edgecolor='k')
sns.distplot(calories['Weightgainedgrams'],hist=False)
plt.boxplot(calories['Weightgainedgrams'])
# CaloriesConsumed
calories['CaloriesConsumed'].skew()
calories['CaloriesConsumed'].kurt()
plt.hist(calories['CaloriesConsumed'],edgecolor='k')
sns.distplot(calories['CaloriesConsumed'],hist=False)
plt.boxplot(calories['CaloriesConsumed'])
# plot scatter diagram
plt.plot(calories.Weightgainedgrams,calories.CaloriesConsumed,"bo");plt.xlabel("Weightgainedgrams");plt.ylabel("CAloriesConsumed")
# findout correlation value
calories.Weightgainedgrams.corr(calories.CaloriesConsumed) 
np.corrcoef(calories.Weightgainedgrams,calories.CaloriesConsumed)
# prepared simple linear regression model for r^2 value
# prepare model 1
model1 = LinearRegression()
model1.fit(calories.Weightgainedgrams.values.reshape(-1,1),calories.CaloriesConsumed)
pred1 = model1.predict(calories.Weightgainedgrams.values.reshape(-1,1))
# Adjusted R-Squared value
model1.score(calories.Weightgainedgrams.values.reshape(-1,1),calories.CaloriesConsumed)
rmse1 = np.sqrt(np.mean((pred1-calories.CaloriesConsumed)**2)) 
model1.coef_
model1.intercept_
# visulization for distribution
plt.scatter(pred1,(pred1-calories.CaloriesConsumed),c="r")
plt.hist(pred1-calories.CaloriesConsumed,color='red',edgecolor='k')
st.probplot(pred1-calories.CaloriesConsumed,dist="norm",plot=pylab)

# prepare model 2
# Fitting Quadratic Regression 
calories["Weightgainedgrams_sqrd"] = calories.Weightgainedgrams*calories.Weightgainedgrams
model2 = LinearRegression()
model2.fit(X = calories.iloc[:,[0,2]],y=calories.CaloriesConsumed)
pred2 = model2.predict(calories.iloc[:,[0,2]])
# Adjusted R-Squared value
model2.score(calories.iloc[:,[0,2]],calories.CaloriesConsumed)
rmse2 = np.sqrt(np.mean((pred2-calories.CaloriesConsumed)**2)) 
model2.coef_
model2.intercept_
# visulization for distribution
plt.scatter(pred2,(pred2-calories.CaloriesConsumed),c="orange")
plt.hist(pred2-calories.CaloriesConsumed,color='orange',edgecolor='k')
st.probplot(pred2-calories.CaloriesConsumed,dist="norm",plot=pylab)

# prepare model 3
# Let us prepare a model by applying transformation on dependent variable
calories["CaloriesConsumed_sqrt"] = np.sqrt(calories.CaloriesConsumed)
model3 = LinearRegression()
model3.fit(X = calories.iloc[:,[0,2]],y=calories.CaloriesConsumed_sqrt)
pred3 = model3.predict(calories.iloc[:,[0,2]])
# Adjusted R-Squared value
model3.score(calories.iloc[:,[0,2]],calories.CaloriesConsumed_sqrt)
rmse3 = np.sqrt(np.mean(((pred3)**2-calories.CaloriesConsumed)**2))
model3.coef_
model3.intercept_
# visulization for distribution
plt.scatter((pred3)**2,((pred3)**2-calories.CaloriesConsumed),c="green")
plt.hist((pred3)**2-calories.CaloriesConsumed,color='green',edgecolor='k')
st.probplot((pred3)**2-calories.CaloriesConsumed,dist="norm",plot=pylab)

# prepare model 4
# Let us prepare a model by applying transformation on dependent variable without transformation on input variables 
model4 = LinearRegression()
model4.fit(X = calories.Weightgainedgrams.values.reshape(-1,1),y=calories.CaloriesConsumed_sqrt)
pred4 = model4.predict(calories.Weightgainedgrams.values.reshape(-1,1))
# Adjusted R-Squared value
model4.score(calories.Weightgainedgrams.values.reshape(-1,1),calories.CaloriesConsumed_sqrt)
rmse4 = np.sqrt(np.mean(((pred4)**2-calories.CaloriesConsumed)**2))
model4.coef_
model4.intercept_
# visulization for distribution
plt.scatter((pred4)**2,((pred4)**2-calories.CaloriesConsumed),c="blue")
plt.hist((pred4)**2-calories.CaloriesConsumed,edgecolor='k')
st.probplot((pred4)**2-calories.CaloriesConsumed,dist="norm",plot=pylab)






































