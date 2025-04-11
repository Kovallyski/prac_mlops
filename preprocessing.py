import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


def pre_preprocess_data(df, metadata=None, test=False):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df = df.drop('Date', axis=1)

    def f(phi):
        return np.sin(phi), np.cos(phi)
    
    dir_dict = {
            'E':   f(0),
            'ENE': f(np.pi / 8),
            'NE':  f(2 * np.pi / 8),
            'NNE': f(3 * np.pi / 8),
            'N':   f(4 * np.pi / 8),
            'NNW': f(5 * np.pi / 8),
            'NW':  f(6 * np.pi / 8),
            'WNW': f(7 * np.pi / 8),
            'W':   f(8 * np.pi / 8),
            'WSW': f(9 * np.pi / 8),
            'SW':  f(10 * np.pi / 8),
            'SSW': f(11 * np.pi / 8),
            'S':   f(12 * np.pi / 8),
            'SSE': f(13 * np.pi / 8),
            'SE':  f(14 * np.pi / 8),
            'ESE': f(15 * np.pi / 8),
    }

    dir_x = {k: v[0] for k, v in dir_dict.items()}
    dir_y = {k: v[1] for k, v in dir_dict.items()}

    for wind_col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
        df[wind_col + '_x'] = df[wind_col].map(dir_x)
        df[wind_col + '_y'] = df[wind_col].map(dir_y)
        # df = df.drop(wind_col, axis=1)
        

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