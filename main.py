from data.get_stock_data import get_stock_data
from features.create_features import create_features
from utils.preprocessing import prepare_data
from models.ann import build_ann
from models.train_model import train_model
from evaluation.evaluate_model import evaluate_model

OHLCV_table = get_stock_data()

features_table = create_features(OHLCV_table)

input_train, input_test, output_train, output_test = prepare_data(features_table)

ann = build_ann(input_train.shape[1])

ann = train_model(ann, input_train, output_train)

evaluate_model(ann, input_test, features_table)
