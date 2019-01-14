# Step1 Data Preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import Datasets
dataset = pd.read_csv('../datasets/50_Startups.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 4 ].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[: , 3] = labelencoder.fit_transform(X[ : , 3])
onehotencoder = OneHotEncoder(categorical_features= [3])

X = X[: , 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

print("X_train")
print(X_train)
print("X_test")
print(X_test)
print("Y_train")
print(Y_train)
print("Y_test")
print(Y_test)

# Step2 train by using multiple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Step3 predict the outcomes
y_pred = regressor.predict(X_test)
print("y_pred")
print(y_pred)


plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.show()

plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.show()