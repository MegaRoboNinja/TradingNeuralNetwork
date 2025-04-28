import numpy as np
import talib
import random

random.seed(42)

# download data of Apple stock
price_AAPL= yf.download('AAPL', start='2017-11-06', end='2023-01-03', auto_adjust = True)

print('importing works!')