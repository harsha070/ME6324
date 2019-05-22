import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import svm

df = pd.read_csv('dataset.csv', delimiter=',', decimal = ',')
df = df.dropna()

print(df.columns)
print(df.corr())

Y = df['% Silica Concentrate']
X = df.drop(['% Silica Concentrate', 'date'], axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.99, random_state = 42)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.30, random_state = 42)

model = svm.SVR(verbose = True)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

error = mean_squared_error(Y_test, Y_pred)

print(error)
