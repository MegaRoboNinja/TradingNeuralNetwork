def train_model(model, input_train, output_train, epochs=100, batch_size=10):
    model.fit(input_train, output_train, batch_size = batch_size, epochs = epochs)