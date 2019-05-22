import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

df = pd.read_csv('dataset.csv', delimiter=',', decimal = ',')
df = df.dropna()

print(df.columns)

Y = df['% Silica Concentrate']
X = df.drop(['% Silica Concentrate', 'date'], axis = 1)
