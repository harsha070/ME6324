import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('dataset.csv', delimiter=',', decimal = ',')
df = df.dropna()

print(df.columns)

Y = df['% Silica Concentrate']
X = df.drop(['% Silica Concentrate', 'date'], axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

for n_trees in range(1,6,1):
    rf = RandomForestRegressor(n_estimators = n_trees, random_state = 42)
    rf.fit(X_train, Y_train)
    predictions = rf.predict(X_test)
    error = mean_squared_error(predictions, Y_test)
    print("Error at ",n_trees," : ",error)
