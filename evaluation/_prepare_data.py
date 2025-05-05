import numpy as np
import pandas as pd

def prepare_trade_data(features_table: pd.DataFrame, output_predicted: np.ndarray) -> tuple[pd.DataFrame,  np.ndarray]:
    print('Preparing trade data...')

    # output_predicted will be a vector of binary values
    # with predictions of whether the stock price will rise or fall
    output_predicted = output_predicted > 0.5

    # prepare the data for the analysis of model predictions
    features_table['output_predicted'] = np.nan
    features_table.iloc[(len(features_table) - len(output_predicted)):,-1:] = output_predicted # fill in the values for the test data

    return features_table.dropna(), output_predicted