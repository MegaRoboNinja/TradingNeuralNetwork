from sklearn.preprocessing import StandardScaler

def prepare_data(features_table):
    input = features_table.iloc[:, 4:-1]
    output = features_table.iloc[:, -1]

    print('Computed the input and expected output values for the model\n')
    print('\nInput data:')
    print(input.shape)
    print(input.iloc[:,0:10])
    print('\nExpected output data for training and testing: (this is a vector o binary values)')
    print(output.shape)
    print(output.iloc[0:10])

    # Split the data into the trainset and testset
    split_index = int(len(features_table)*0.8)
    input_train, input_test, output_train, output_test = input[:split_index], input[split_index:], output[:split_index], output[split_index:]

    print('\nDivided into test set and training set at index ', split_index, '\n')

    # DATA PREPROCESSING – Standarise the dataset
    # Ensure that there is no bias associated with diffrent scales of the input features
    # Transform the input so that for all features the mean is equal to 0 and variance to 1
    # The output values contain binary values hence they need not be standarised
    # -------------------------------------------------------------------------------------------

    # Feature Scaling
    sc = StandardScaler()
    input_train = sc.fit_transform(input_train)
    input_test = sc.transform(input_test)

    print('Standarized the dataset')

    return input_train, input_test, output_train, output_test