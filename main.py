import numpy as np
import pandas as pd
import talib
import random
import yfinance as yf
import tensorflow as tf

random.seed(42)

# DETTING DATA 
# ------------------------------------------------------------------------------------------

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

# Split the data into the trainset and testset
split = int(len(price_AAPL)*0.8)
input_train, input_test, output_train, output_test = input[:split], input[split:], output[:split], output[split:]

print('\nDivided into test set and training set at index ', split, '\n')

# DATA PREPROCESSING – Standarise the dataset
# Ensure that there is no bias associated with diffrent scales of the input features
# Transform the input so that for all features the mean is equal to 0 and variance to 1
# The output values contain binary values hence they need not be standarised
# -------------------------------------------------------------------------------------------

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
input_train = sc.fit_transform(input_train)
input_test = sc.transform(input_test)

print('Standarized the dataset')

# BUILD THE ARTIFICIAL NEURAL NETWORK
# -------------------------------------------------------------------------------------------
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

print('Building the artificial neural network...')

# Sequentially build the layers forming the perceptron
classifier = Sequential()

# 128 neurons a layer
# uniform initializer - the initial values of the neurons are uniform
# the first layer after needs the input dimension
# following layers automaticly get input dimension from their preceding layer
classifier.add(Dense(units = 128, kernel_initializer = 'uniform',
                      activation = 'relu', input_dim = input.shape[1]))
classifier.add(Dense(units = 128, kernel_initializer = 'uniform',
                      activation = 'relu'))
# the output layer - a single neuron with sigmoid activation function
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
print('done')

# Compiling the classifier
# Determinig how the model will be trained
# Defining the optimization algorithm, cost function and metrics
# (metrics do not affect training - they are just for monitoring the progress)
print('Compiling the classifier...')
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
print('done')