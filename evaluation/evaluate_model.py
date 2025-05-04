import numpy as np
from sklearn.metrics import accuracy_score

def evaluate_model(model, input_test, output_test, features_table):
    print('Calculating the returns...')

    # output_predicted will be a vector of binary values
    # with predictions of whether the stock price will rise or fall
    output_predicted = model.predict(input_test)
    output_predicted = output_predicted > 0.5

    # prepare the data for the analysis of model predictions
    features_table['output_predicted'] = np.nan
    features_table.iloc[(len(features_table) - len(output_predicted)):,-1:] = output_predicted # fill in the values for the test data
    trade_price = features_table.dropna()

    # Accuracy
    accuracy = accuracy_score(output_test, output_predicted)
    print(f"Model accuracy (price will rise?): {accuracy:.2%}")

    # Computing Strategy Returns
    trade_price['Tomorrows Returns'] = 0.
    trade_price['Tomorrows Returns'] = np.log(trade_price['Close']/trade_price['Close'].shift(1))
    trade_price['Tomorrows Returns'] = trade_price['Tomorrows Returns'].shift(-1)

    trade_price['Strategy Returns'] = 0.
    trade_price['Strategy Returns'] = np.where(trade_price['output_predicted'] == True, 
                                                    trade_price['Tomorrows Returns'], - trade_price['Tomorrows Returns'])

    trade_price['Cumulative Market Returns'] = np.cumsum(trade_price['Tomorrows Returns'])
    trade_price['Cumulative Strategy Returns'] = np.cumsum(trade_price['Strategy Returns'])

    print('done')

    # Plotting the graph of returns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.plot(trade_price['Cumulative Market Returns'], color='r', label='Market Returns')
    plt.plot(trade_price['Cumulative Strategy Returns'], color='g', label='Strategy Returns')

    plt.title('Market returns and Strategy returns', color='purple', size=15)

    # Setting axes labels for close prices plot
    plt.xlabel('Dates', {'color': 'orange', 'fontsize':15})
    plt.ylabel('Returns(%)', {'color': 'orange', 'fontsize':15})

    plt.legend()
    plt.show()


    # Ostatnia wartość skumulowanych strategii zwrotów
    final_cum_return = trade_price['Cumulative Strategy Returns'].iloc[-2]
    # Zamiana logarytmicznego zwrotu na procentowy
    total_return_percentage = (np.exp(final_cum_return) - 1) * 100
    print(total_return_percentage)

    print(f"Total % return in the test period: {total_return_percentage:.2f}%")