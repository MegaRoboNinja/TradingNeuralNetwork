import talib
import numpy as np

def create_features(OHLCV_table):
    # Creating input features – bit more processed data, that the model will train on
    OHLCV_table['H-L'] = OHLCV_table['High'] - OHLCV_table['Low']
    OHLCV_table['O-C'] = OHLCV_table['Close'] - OHLCV_table['Open']
    OHLCV_table['3day MA'] = OHLCV_table['Close'].shift(1).rolling(window = 3).mean()
    OHLCV_table['10day MA'] = OHLCV_table['Close'].shift(1).rolling(window = 10).mean()
    OHLCV_table['30day MA'] = OHLCV_table['Close'].shift(1).rolling(window = 30).mean()
    OHLCV_table['Std_dev']= OHLCV_table['Close'].rolling(5).std()
    OHLCV_table['RSI'] = talib.RSI(OHLCV_table['Close'].values, timeperiod = 9)
    OHLCV_table['Williams %R'] = talib.WILLR(OHLCV_table['High'].values, OHLCV_table['Low'].values, OHLCV_table['Close'].values, 7)
    # define Price_Rise that is equivalent to our output value and what it will be tested againts
    OHLCV_table['Price_Rise'] = np.where(OHLCV_table['Close'].shift(-1) > OHLCV_table['Close'], 1, 0)

    # at this point the OHLCV table is extended with all the features we added
    return OHLCV_table.dropna()

     