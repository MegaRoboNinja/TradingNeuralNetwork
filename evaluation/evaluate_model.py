import numpy as np
from sklearn.metrics import accuracy_score

from evaluation._prepare_data import prepare_trade_data
from evaluation._calculate_returns import compute_strategy_returns
from evaluation._plot_returns import plot_returns

# Punkt wejścia do ewaluacji i testowania zarządzający poszczególnymi testami i oceną
def evaluate_model(model, input_test, output_test, features_table):
    print('Evaluating the trained model...')

    # Use the trained model to predict outcomes for the input_test features
    output_predicted = model.predict(input_test)

    trade_price, output_predicted = prepare_trade_data(features_table, output_predicted)

    accuracy = accuracy_score(output_test, output_predicted)
    print(f"Model accuracy (price will rise?): {accuracy:.2%}")

    compute_strategy_returns(trade_price)

    plot_returns(trade_price)

    # Final return
    final_cum_return = trade_price['Cumulative Strategy Returns'].iloc[-2]
    # Logarithmic to percentage
    total_return_percentage = (np.exp(final_cum_return) - 1) * 100
    print(f"Total % return in the test period: {total_return_percentage:.2f}%")