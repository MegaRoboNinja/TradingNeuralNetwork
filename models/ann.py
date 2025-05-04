from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def build_ann(input_dim):
    print('Building the artificial neural network...')

    # Sequentially build the layers forming the perceptron
    classifier = Sequential()

    # 128 neurons a layer
    # uniform initializer - the initial values of the neurons are uniform
    # the first layer after needs the input dimension
    # following layers automaticly get input dimension from their preceding layer
    classifier.add(Dense(units = 128, kernel_initializer = 'uniform',
                        activation = 'relu', input_dim=input_dim))
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