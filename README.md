# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
Load the dataset and separate input features (X) and target variable (car price y).
Split the dataset into training and testing sets.
Train Linear Regression and Polynomial Regression models using the training data.
Predict car prices using both models and compare their performance. 
```
## Program:
```
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: Roopika m
RegisterNumber: 212225040348
*/
```

```
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

df=pd.read_csv('CarPrice_Assignment.csv')
data= df.drop(['car_ID', 'CarName'], axis=1)
data = pd.get_dummies(data, drop_first=True)
X=data.drop('price', axis=1)
y=data['price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)
model= LinearRegression()
model.fit(X_train,y_train)
LinearRegression
LinearRegression()
print("Name:Roopika m")
print("Reg. No:25008774")

print("\n=== Cross-Validation ===")
cv_scores = cross_val_score(model, X, y, cv=5)

print("Fold R2 scores:", [f"{score:.4f}" for score in cv_scores])
print(f"Average R2: {cv_scores.mean():.4f}")

y_pred = model.predict(X_test)

print("\n=== Test Set Performance ===")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"MAE:{mean_absolute_error(y_test,y_pred):.2f}")
Name:roopika m
Reg. No:212225040348

=== Cross-Validation ===
Fold R2 scores: ['0.6238', '0.6316', '0.3132', '0.3643', '-0.4944']
Average R2: 0.2877

=== Test Set Performance ===
MSE: 8482008.48
R²: 0.8926
MAE:2089.38
Name:roopika m
Reg. No:212225040348

=== Cross-Validation ===
Fold R2 scores: ['0.6238', '0.6316', '0.3132', '0.3643', '-0.4944']
Average R2: 0.2877

=== Test Set Performance ===
MSE: 8482008.48
R²: 0.8926
MAE:2089.38

*/
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: 
RegisterNumber:  
*/
```

## Output:
```
Name: Roopika m
Reg. No: 25008774
Linear Regression:
MSE= 16471505.900042146
R2-Score= 0.7913520781370976

Polynomial Regression:
MSE: 15247661.89
R^2: 0.81
MAE:2694.05
```
![simple linear regression model for predicting the marks scored](sam.png)
<img width="868" height="468" alt="image" src="https://github.com/user-attachments/assets/393a47ed-ae18-4dfe-8a6f-cc3d492fdbc5" />

## Result:

Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
