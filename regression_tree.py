import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import tree

df = pd.read_csv('dataset.csv', delimiter=',', decimal = ',')
df = df.dropna()

print(df.columns)

Y = df['% Silica Concentrate']
X = df.drop(['% Silica Concentrate', 'date'], axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

model = tree.DecisionTreeRegressor()

model = model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

error = mean_squared_error(Y_pred, Y_test)
print(error)

accuracy = 0.0

Y_test = Y_test.values
n = len(Y_test)

for i in range(n):
    if(abs(Y_test[i]-Y_pred[i]) <= 0.05 * Y_test[i]):
        accuracy = accuracy + 1

print(accuracy/n)
