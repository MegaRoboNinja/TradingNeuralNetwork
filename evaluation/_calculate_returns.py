import pandas as pd
import numpy as np

def compute_strategy_returns(trade_price: pd.DataFrame):
    print('Calculating returns...')

    trade_price['Tomorrows Returns'] = 0.
    trade_price['Tomorrows Returns'] = np.log(trade_price['Close']/trade_price['Close'].shift(1))
    trade_price['Tomorrows Returns'] = trade_price['Tomorrows Returns'].shift(-1)

    trade_price['Strategy Returns'] = 0.
    trade_price['Strategy Returns'] = np.where(trade_price['output_predicted'] == True, 
                                                    trade_price['Tomorrows Returns'], - trade_price['Tomorrows Returns'])

    trade_price['Cumulative Market Returns'] = np.cumsum(trade_price['Tomorrows Returns'])
    trade_price['Cumulative Strategy Returns'] = np.cumsum(trade_price['Strategy Returns'])