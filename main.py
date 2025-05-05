import argparse
from data.get_stock_data import get_stock_data
from features.create_features import create_features
from model_manager import get_or_train_model
from utils.preprocessing import prepare_data
from models.ann import build_ann
from models.train_model import train_model
from evaluation.evaluate_model import evaluate_model

def main(model_path):
    OHLCV_table = get_stock_data()

    features_table = create_features(OHLCV_table)

    input_train, input_test, output_train, output_test = prepare_data(features_table)

    ann = get_or_train_model(model_path, input_train, output_train, 100, 10)

    evaluate_model(ann, input_test, output_test, features_table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to save or load model')
    args = parser.parse_args()
    main(args.model)