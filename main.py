import numpy as np
import pandas as pd
import talib
import random
import yfinance as yf

random.seed(42)

# download data of Apple stock
price_AAPL= yf.download('AAPL', start='2017-11-06', end='2023-01-03', auto_adjust = True)

print('\nDownloaded data of Apple Stock')
print(price_AAPL.shape)
print(price_AAPL.columns)

# Preparing the dataset
# Flattten the hierarchical multiindex structure as we only have one index 'AAPL'
price_AAPL.columns = price_AAPL.columns.get_level_values(0)

print('\nFlattended the hierarchical multiindex dataframe structure')
print(price_AAPL.shape)
print(price_AAPL.columns)

price_AAPL = price_AAPL.dropna()
price_AAPL = price_AAPL.reset_index()  # Flatten the index

# Creating input features – bit more processed data, that the model will train on
price_AAPL['H-L'] = price_AAPL['High'] - price_AAPL['Low']
price_AAPL['O-C'] = price_AAPL['Close'] - price_AAPL['Open']
price_AAPL['3day MA'] = price_AAPL['Close'].shift(1).rolling(window = 3).mean()
price_AAPL['10day MA'] = price_AAPL['Close'].shift(1).rolling(window = 10).mean()
price_AAPL['30day MA'] = price_AAPL['Close'].shift(1).rolling(window = 30).mean()
price_AAPL['Std_dev']= price_AAPL['Close'].rolling(5).std()
price_AAPL['RSI'] = talib.RSI(price_AAPL['Close'].values, timeperiod = 9)
price_AAPL['Williams %R'] = talib.WILLR(price_AAPL['High'].values, price_AAPL['Low'].values, price_AAPL['Close'].values, 7)
# define Price_Rise that is equivalent to our output value on what it will be tested againts
price_AAPL['Price_Rise'] = np.where(price_AAPL['Close'].shift(-1) > price_AAPL['Close'], 1, 0)

input = price_AAPL.iloc[:, 4:-1]
output = price_AAPL.iloc[:, -1]

print('Computed the input and expected output values for the model\n')
print('\nInput data:')
print(input.shape)
print(input.iloc[:,0:10])
print('\nExpected output data for training and testing: (this is a vector o binary values)')
print(output.shape)
print(output.iloc[0:10])


# 
