#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 14:13:51 2025

@author: sofi
"""
# https://blog.quantinsti.com/neural-network-python/
import numpy as np
import pandas as pd
import talib
import random
import yfinance as yf
# Standardizing
from sklearn.preprocessing import StandardScaler
# ANN
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import Dropout
# plotting
import matplotlib.pyplot as plt
random.seed(42)
# Fetching data
price_AAPL = yf.download('AAPL', start='2017-11-06', end='2023-01-03',auto_adjust = True)
#Preparing the dataset
"""
Our Features:
    ----------
High minus Low price
Close minus Open price
Three day moving average
Ten day moving average
30 day moving average
Standard deviation for a period of 5 days
Relative Strength Index
Williams %R
"""
price_AAPL['H-L'] = price_AAPL['High'] - price_AAPL['Low']
price_AAPL['O-C'] = price_AAPL['Close'] - price_AAPL['Open']
price_AAPL['3day MA'] = price_AAPL['Close'].shift(1).rolling(window=3).mean()
price_AAPL['10day MA'] = price_AAPL['Close'].shift(1).rolling(window=10).mean()
price_AAPL['30day MA'] = price_AAPL['Close'].shift(1).rolling(window=30).mean()
price_AAPL['Std_dev'] = price_AAPL['Close'].rolling(5).std()
price_AAPL['RSI'] = talib.RSI(price_AAPL['Close'].values, timeperiod=9)
price_AAPL['Williams %R'] = talib.WILLR(price_AAPL['High'].values, price_AAPL['Low'].values, price_AAPL['Close'].values, 7)
# Defining the output ( 1 if the closing price of tomorrow is greater than the closing price of today)
price_AAPL['Price_Rise'] = np.where(price_AAPL['Close'].shift(-1)>price_AAPL['Close'],1,0)
# Dropping the rows that store NaN values
price_AAPL = price_AAPL.dropna()
# Creating data frames to storethe input (X) and the output variables (Y)

X = price_AAPL.iloc[:,4:-1]
y = price_AAPL.iloc[:,-1]
# Splitting the dataset into training and validation(test) datasets
split = int(len(price_AAPL)*0.8)
X_train, X_test, y_train, y_test = X[:split],X[split:], y[:split],y[split:]
# Standardizing the dataset -> no need to standardize y, binary values
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transfort(X_test)
# Building the Artificial Neural Network
classifier = Sequential()
classifier.add(Dense(units=128, kernel_initializer="uniform", activation="relu",input_dim = X.shape[1]))
classifier.add(Dense(units=128, kernel_initializer="uniform",activation="relu"))
classifier.add(Dense(units=1, kernel_initializer="uniform",activation="sigmoid"))
#Compiling the classifier + fitting the model
classifier.compile(optimizer="adam",loss="mean_squared_error", metrics=['accuracy'])
classifier.fit(X_train,y_train, batch_size=10,epocs=100)
# Predicting
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)
#
price_AAPL['y_pred'] = np.NaN
price_AAPL.iloc[(len(price_AAPL) - len(y_pred)):,-1:]=y_pred
trade_price_APPL = price_AAPL.dropna()
# Taking a long position : predicted value is true
# Taking a short position : predicted value is false
# Computing Strategy Returns

trade_price_APPL['Tommorows Returns'] = 0.
trade_price_APPL['Tomorrows Returns'] = np.log(trade_price_APPL['Close']/trade_price_APPL['Close'].shift(1))
trade_price_APPL['Tomorrows Returns'] = trade_price_APPL['Tomorrows Returns'].shift(-1)

trade_price_APPL['Strategy Returns'] = 0.
trade_price_APPL['Strategy Returns'] = np.where(trade_price_APPL['y_pred']==True, trade_price_APPL['Tomorrow Returns'], -trade_price_APPL['Tomorrow returns'])

trade_price_APPL['Cumulative Market Returns'] = np.cumsum(trade_price_APPL['Tomorrow Returns'])
trade_price_APPL['Cumulative Strategy Returns'] = np.cumsum(trade_price_APPL['Strategy returns'])

plt.figure(figsize = (10,5))
plt.plot(trade_price_APPL['Cumulative Market Returns'], color='r', label='Market Returns')
plt.plot(trade_price_APPL['Cumulative Strategy Returns'], color='g',  label = 'Strategy Returns')
plt.title('Market returns and Strategy returns', color='purple', size=15)
#Setting axes labels for close price plot

plt.xlabel('Dates', {'color':'orange', 'fontsize': 15})
plt.ylabel('Returns (%)', {'color':'orange','fontsize':15})

plt.legend()
plt.show()
