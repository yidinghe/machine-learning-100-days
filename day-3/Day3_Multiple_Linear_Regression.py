# Step1 Data Preprocessing
import pandas as pd
import numpy as np

dataset = pd.read_csv('../datasets/50_Startups.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 4 ].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LabelEncoder