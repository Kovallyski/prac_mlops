import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


PREPROC_PATH = 'preproc'

def pre_preprocess_data(df, metadata=None, test=False):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df = df.drop('Date', axis=1)

    print(df.columns)

    if not test and 'RainTomorrow' in df.columns:
        df = df.dropna(subset=['RainTomorrow'])
        return df.drop('RainTomorrow', axis=1), df['RainTomorrow'].apply(lambda x: x == 'Yes')
    else:
        return df

def create_preprocessor(X, metadata=None, **kwargs):
    return make_pipeline(
            ColumnTransformer(
                [
                    #('location', OneHotEncoder(dtype='int', handle_unknown='ignore'), ['Location']),
                    # ('gust_dir', OneHotEncoder(dtype='int'), ['WindGustDir']),
                    # ('9am_dir', OneHotEncoder(dtype='int'), ['WindDir9am']),
                    # ('3pm_dir', OneHotEncoder(dtype='int'), ['WindDir3pm']),
                    ('rain_today', OneHotEncoder(dtype='int', handle_unknown='ignore', drop='if_binary'), ['RainToday']),
                    ('scaler', StandardScaler(), list(X.select_dtypes(include='number').columns))
                ],
            ),
            SimpleImputer()
        )