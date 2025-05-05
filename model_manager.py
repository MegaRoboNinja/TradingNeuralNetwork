import os
from tensorflow.keras.models import load_model
from models.ann import build_ann
from models.train_model import train_model

def get_or_train_model(model_path, input_train, output_train, epochs=100, batch_size=10):
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        return load_model(model_path)
    else:
        print("Training new model...")
        model = build_ann(input_train.shape[1])
        train_model(model, input_train, output_train, epochs, batch_size)
        model.save(model_path)
        print(f"Model saved to {model_path}")
        return model