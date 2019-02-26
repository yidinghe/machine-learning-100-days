# Step1 Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Datasets
dataset = pd.read_csv('../datasets/Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values
print('X')
print(X)
print('Y')
print(Y)

# Separate TrainSets and TestSets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


