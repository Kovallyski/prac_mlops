import argparse
from ast import parse
import json
import pandas as pd
import numpy as np
import pickle
import os
import sklearn
from sklearn.pipeline import Pipeline

from src.database import DataBase
from src.eda import check_data_quality
from src.preprocessing import create_preprocessor, pre_preprocess_data
import src.preprocessing as preprocessing
from src.train import ModelTrainer
from src.utils import display

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, choices=['inference', 'update', 'summary', 'add_data'], help="Run mode")
    parser.add_argument("-f", "--file_path", type=str, help="File path", default='')
    parser.add_argument("-r", "--retrain", action='store_true', help="Retrain models by default")

    return parser


def run_model(db, model_type, args, file_path):

    mt = ModelTrainer(f"cfg/{model_type}.json")
    model_name = mt.load_best_model()
    with open(os.path.join(preprocessing.PREPROC_PATH, model_name), 'rb') as f:
        preproc = pickle.load(f)

    df, metadata = db.load_test(file_path)
    df, metadata, _ = check_data_quality(df, metadata, metadata)
    X = pre_preprocess_data(df, metadata, test=True)
    X = preproc.transform(X)
    preds = mt.predict(X)
    
    df = pd.read_csv(file_path)
    df['predict'] = [['No', 'Yes'][x] for x in preds]
    save_path = file_path[:-4] + '_with_predictions.csv'
    df.to_csv(save_path, index=False)
    print(f'Saved predictions in {save_path}!')


def inference(file_path, **kwargs):
    with open("cfg/config.json", 'r') as f:
        db = DataBase(json.load(f))

    run_model(db, 'random_forest', args, file_path)
    run_model(db, 'logistic', args, file_path)
    run_model(db, 'knn', args, file_path)


# NOTE: note all implemented functions are demostrated
def up_one(db, model_type, args):

    mt = ModelTrainer(f"cfg/{model_type}.json")
    model_name = mt.load_best_model()

    mode = 0
    if not args.retrain and db.get_known():
        df_new, metadata_new, _, metadata_old = db.get_data(db.get_unknown())
        df_new, metadata_new, score = check_data_quality(df_new, metadata_new, metadata_old)

        X_new, y_new = pre_preprocess_data(df_new, metadata_old) 
        with open(os.path.join(preprocessing.PREPROC_PATH, model_name), 'rb') as f:
            preproc = pickle.load(f)
        X_new = preproc.transform(X_new)

        display(f'New data score: {score}')
        
        if score < 0.9:
            display(f'DATA DRIFT DETECTED!')
            display(f'Retuning model and preprocessors on all data')
            mode = 2

        elif mt.detect_model_drift(X_new, y_new):
            display(f'MODEL DRIFT DETECTED!')
            display(f'Retuning model and preprocessors on all data')
            mode = 1
        else:
            mode = 0
            display(f'Continuing model training')

    else:
        mode = 2
        print("Training model from scratch!")

    if mode == 0:
        # X_new should be reloaded with new metadata, but is not since metadata merge without a full load is currently impossible
        _, metr = mt.update_model(X_new, y_new)
        mt.save_model(metr, best=True)
    else:
        df, metadata, _, _ = db.get_data()
        df, metadata, score = check_data_quality(df, metadata, metadata)
        X, y = pre_preprocess_data(df, metadata)
        preproc = create_preprocessor(X, metadata)

        if mode == 1:
            X = preproc.fit_transform(X)
            _, _, model_name = mt.hyperparameter_tuning(X, y, val='TSS')
        else: # mode == 2
            _, _, preproc, model_name = mt.hyperparameter_tuning_with_preproc(X, y, preproc, val='TSS')
            preproc = Pipeline(steps=preproc).fit(X)

        print(model_name)
        with open(os.path.join(preprocessing.PREPROC_PATH, model_name), 'wb') as f:
            pickle.dump(preproc, f)

    print('Updated model!')

def update(**kwargs):
    with open("cfg/config.json", 'r') as f:
        db = DataBase(json.load(f))
        db.load_train()

    if not db.get_unknown():
        print("No new data to update models on!")
        return

    up_one(db, 'random_forest', args)
    up_one(db, 'logistic', args)
    up_one(db, 'knn', args)
    db.set_known()

def get_sum(X, y, model_type, **kwargs):
    mt = ModelTrainer(f"cfg/{model_type}.json")
    model_name = mt.load_best_model()
    with open(os.path.join(preprocessing.PREPROC_PATH, model_name), 'rb') as f:
        preproc = pickle.load(f)
    X = preproc.transform(X)

    mt.feature_names = preproc.get_feature_names_out()
    mt.class_names = ['No', 'Yes']
    print("Saved model summary in", mt.generate_summary())
    mt.interpret_model(X, y)
    
def summary(**kwargs):
    with open("cfg/config.json", 'r') as f:
        db = DataBase(json.load(f))

    df, metadata, _, quantile = db.get_data()
    df, metadata, score = check_data_quality(df, metadata, quantile)
    X, y = pre_preprocess_data(df, metadata)

    get_sum("random_forest")
    get_sum('logistic')
    get_sum("knn")

# NOTE: note all implemented functions are demostrated
def add_data(file_path, **kwargs):
    with open("cfg/config.json", 'r') as f:
        db = DataBase(json.load(f))
    batch_id = db.load_train(file_path)

def main(args):
    os.makedirs('preproc', exist_ok=True)
    eval(f'{args.mode}(file_path=args.file_path)')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
