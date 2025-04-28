import numpy as np
import pandas as pd
import talib
import random
import yfinance

random.seed(42)

# download data of Apple stock
price_AAPL= yf.download('AAPL', start='2017-11-06', end='2023-01-03', auto_adjust = True)

# Preparing the dataset
# Creating input features – bit more processed data, that the model will train on
price_AAPL['H-L'] = price_AAPL['High'] - price_AAPL['Low']
price_AAPL['O-C'] = price_AAPL['Close'] - price_AAPL['Open']
price_AAPL['3day MA'] = price_AAPL['Close'].shift(1).rolling(window = 3).mean()
price_AAPL['10day MA'] = price_AAPL['Close'].shift(1).rolling(window = 10).mean()
price_AAPL['30day MA'] = price_AAPL['Close'].shift(1).rolling(window = 30).mean()
price_AAPL['Std_dev']= price_AAPL['Close'].rolling(5).std()
price_AAPL['RSI'] = talib.RSI(price_AAPL['Close'].values, timeperiod = 9)
price_AAPL['Williams %R'] = talib.WILLR(dataset['High'].values, price_AAPL['Low'].values, price_AAPL['Close'].values, 7)
# define Price_Rise that is equivalent to our output value on what it will be tested againts
price_AAPL['Price_Rise'] = np.where(price_AAPL['Close'].shift(-1) > price_AAPL['Close'], 1, 0)

price_AAPL = price_AAPL.dropna()

input = price_AAPL.iloc[:, 4:-1]
output = price_AAPL.iloc[:, -1]