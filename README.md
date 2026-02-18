# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: Rubika
RegisterNumber: 212225040348
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
df=pd.read_csv('CarPrice_Assignment.csv')
df.head
X=df[['enginesize','horsepower','citympg','highwaympg']]
Y=df['price']
df.head()
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
#feature scaling
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
#train model
model=LinearRegression()
model.fit(X_train_scaled,Y_train)
#prediction
Y_pred=model.predict(X_test_scaled)
print("Name:Rubika")
print("Reg. No:212225040348")
print("MODEL COEFFICIENTS:")
for feature,coef in zip(X.columns,model.coef_):
    print(f"{feature:>12}: {coef:>10}")
print(f"{'Intercept':>12}: {model.intercept_:>10}")
print("\nMODEL PERFORMANCE:")
print(f"{'MSE':>12}: {mean_squared_error(Y_test,Y_pred):>10}")
print(f"{'RMSE':>12}: {np.sqrt(mean_squared_error(Y_test,Y_pred)):>10}")
print(f"{'R-squared':>12}: {r2_score(Y_test,Y_pred):10}")
print(f"{'MAE':>12}: {mean_absolute_error(Y_test,Y_pred):10}")
# linearity check
plt.figure(figsize=(10,5))
plt.scatter(Y_test,Y_pred,alpha=0.6)
plt.plot([Y.min(),Y.max()],[Y.min(),Y.max()],'r--')
plt.title("Linearity Check: Actual vs Predicted Price")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()
# Independence (Durbin-watson)
residuals=Y_test-Y_pred
dw_test=sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson statistic: {dw_test:.2f}",
      "\n(Values close to 2 indicates no autocorrelation)")
# Homoscedasticity
plt.figure(figsize=(10,5))
sns.residplot(x=Y_pred,y=residuals,lowess=True,line_kws={'color':'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()
# Normality of residuals
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
sns.histplot(residuals,kde=True,ax=ax1)
ax1.set_title("Residuals Distrubution")
sm.qqplot(residuals,line='45',fit=True,ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: 
RegisterNumber:  
*/
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: 
RegisterNumber:  
*/
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
<img width="868" height="468" alt="image" src="https://github.com/user-attachments/assets/3abbb65b-f102-4f71-bd74-c097221a1a35" />


## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
