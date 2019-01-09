#Step 1: Data Preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('../datasets/studentscores.csv')
X = dataset.iloc[:, : 1].values
Y = dataset.iloc[:, 1 ].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/4, random_state = 0)
print('X_train')
print(X_train)
print('X_test')
print(X_test)
print('Y_train')
print(Y_train)
print('Y_test')
print(Y_test)

#Step 2: LinearRegression 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

#Step 3: Prediction Outcome
Y_pred = regressor.predict(X_test)
print('Y_pred')
print(Y_pred)

#Step 4: Visulization
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.show()
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.show()
