import yfinance as yf

def get_stock_data():
    OHLCV_table = yf.download('AAPL', start='2017-11-06', end='2025-05-03', auto_adjust = True)
    print('\nDownloaded the OHLCV table')
    print(OHLCV_table.shape)
    print(OHLCV_table.columns)
    # Preparing the dataset
    # Flattten the hierarchical multiindex structure as we only have one index 'AAPL'
    OHLCV_table.columns = OHLCV_table.columns.get_level_values(0)

    print('\nFlattended the hierarchical multiindex dataframe structure')
    print(OHLCV_table.shape)
    print(OHLCV_table.columns)
    return OHLCV_table


