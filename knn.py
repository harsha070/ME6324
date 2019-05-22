import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import time

df = pd.read_csv('dataset.csv', delimiter=',', decimal = ',')
df = df.dropna()

print(df.columns)

Y = df['% Silica Concentrate']
X = df.drop(['% Silica Concentrate', 'date'], axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

start_time = time.time()

model = KNeighborsRegressor(n_neighbors = 5)

model.fit(X_train, Y_train)

end_time = time.time()

print("Time elapsed: ",end_time - start_time)

Y_pred = model.predict(X_test)

print("Time elapsed: ",time.time()-end_time)

error = mean_squared_error(Y_pred, Y_test)

print(error)
