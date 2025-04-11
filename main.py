import argparse
from ast import parse
import json
import pandas as pd
import numpy as np
import pickle
import os
import sklearn

from database import DataBase
from eda import check_data_quality
from preprocessing import create_preprocessor, pre_preprocess_data
from train import ModelTrainer
from utils import display

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, choices=['inference', 'update', 'summary', 'add_data'], help="Run mode")
    parser.add_argument("-f", "--file_path", type=str, help="File path", default='')

    return parser


def inference(file_path, **kwargs):
    with open("configure.json", 'r') as f:
        db = DataBase(json.load(f))

    # TODO: should actually store the preprocessor in the model pipeline
    df, metadata, quantile = db.get_data()
    X, y = pre_preprocess_data(df, metadata)
    preproc = create_preprocessor(X, metadata).fit(X)

    df, metadata = db.load_test(file_path)
    X = pre_preprocess_data(df, metadata, test=True)
    X = preproc.transform(X)

    mt = ModelTrainer()
    mt.load_best_model()
    preds = mt.predict(X)


    df = pd.read_csv(file_path)
    df['predict'] = [['No', 'Yes'][x] for x in preds]
    save_path = file_path[:-4] + '_with_predictions.csv'
    df.to_csv(save_path, index=False)
    print(f'Saved predictions in {save_path}!')


# NOTE: note all implemented functions are demostrated
def update(**kwargs):
    with open("configure.json", 'r') as f:
        db = DataBase(json.load(f))

    # For now doesn't get only new data, since models checkpointing will probably stop making sense
    df, metadata, quantile = db.get_data()
    df, metadata, score = check_data_quality(df, metadata, quantile)
    display(f'SCORE: {score}')

    X, y = pre_preprocess_data(df, metadata)
    preproc = create_preprocessor(X, metadata)

    mt = ModelTrainer()
    mt.load_best_model()
    # Dummy method for now, chooses best preprocessor, but doesn't actually save it as part of the model
    mt.hyperparameter_tuning_with_preproc(X, y, preproc) 
    print('Updated model!')

def summary(**kwargs):
    with open("configure.json", 'r') as f:
        db = DataBase(json.load(f))

    df, metadata, quantile = db.get_data()
    df, metadata, score = check_data_quality(df, metadata, quantile)
    display(f'SCORE: {score}')

    X, y = pre_preprocess_data(df, metadata)
    preproc = create_preprocessor(X, metadata)
    X = preproc.fit_transform(X)

    mt = ModelTrainer()
    mt.load_best_model()
    mt.feature_names = preproc.get_feature_names_out()
    mt.class_names = ['No', 'Yes']
    print("Saved model summary in", mt.generate_summary())
    mt.interpret_model(X, y)

# NOTE: note all implemented functions are demostrated
def add_data(file_path, **kwargs):
    with open("configure.json", 'r') as f:
        db = DataBase(json.load(f))
    batch_id = db.load_train(file_path)
    db.set_known(batchids=[batch_id])


def main(args):
    eval(f'{args.mode}(file_path=args.file_path)')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)