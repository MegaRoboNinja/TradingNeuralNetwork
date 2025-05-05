import pandas as pd
import matplotlib.pyplot as plt

def plot_returns(trade_price: pd.DataFrame):
    print('Plotting returns...')

    # Plotting the graph of returns
    plt.figure(figsize=(10,5))
    plt.plot(trade_price['Cumulative Market Returns'], color='r', label='Market Returns')
    plt.plot(trade_price['Cumulative Strategy Returns'], color='g', label='Strategy Returns')

    plt.title('Market returns and Strategy returns', color='purple', size=15)

    # Setting axes labels for close prices plot
    plt.xlabel('Dates', {'color': 'orange', 'fontsize':15})
    plt.ylabel('Returns(%)', {'color': 'orange', 'fontsize':15})

    plt.legend()
    plt.show()