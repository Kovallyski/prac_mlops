from numpy import median
import numpy as np
import pandas as pd


def check_data_quality(df, metadata_df, quantile_df):
    category_columns = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
    target_columns = ['RainTomorrow']
    numeric_columns = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
        'Sunshine', 'WindGustSpeed',
        'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
        'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
        'Temp3pm']
    datetime_columns = ['Date']
    index_columns = ['Date', 'Location']

    metadata_df = metadata_df.set_index(['type'])
    quantile_df = quantile_df.set_index(['type'])


    full_duplicated = df.duplicated()
    collisions = df[index_columns].duplicated(keep=False) & ~df.duplicated(keep=False)
    print(f'Detected {full_duplicated.sum()} duplicates and {collisions.sum()} collisions')
    if full_duplicated.sum() + collisions.sum():
        df = df.drop(df[full_duplicated].index)
        df = df.drop(df[collisions].index)
        print(f'All duplicates and collisions removed')

    new_datetime = pd.to_datetime(df['Date'], errors='coerce')
    bad_values = new_datetime.isna()
    if bad_values.sum():
        print(f"Found values {list(df['Date'][bad_values].unique())} out of range in column Date", end='')
        df['Date'] = new_datetime
        df = df.drop(df[bad_values].index)
        print(', all removed.')


    def check_winddir(s):
        return s.isin(['E', 'SSW', 'NE', 'ENE', 'SE', 'SSE', 'WNW', 'N', 'NW', 'W', 'S', 'SW', 'WSW', 'NNW', 'ESE', 'NNE']) | s.isna()

    def angle(d):
        if 'E' in d:
            return (d.count('E') * 0 + d.count('N') * np.pi/2  - d.count('S') * np.pi / 2) / len(d)
        if 'W' in d:
            return (d.count('W') * np.pi + d.count('N') * np.pi/2  - d.count('S') * np.pi * 3 / 2) / len(d)
        return (d.count('N') * np.pi/2  - d.count('S') * np.pi / 2) / len(d)

    for wind_dir in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
        bad_values = df[wind_dir][~check_winddir(df[wind_dir])]
        if bad_values.sum():
            print(f"Found values {list(bad_values.unique())} out of range in column {wind_dir}, all removed")
            df.loc[bad_values, wind_dir] = pd.NA
        df[wind_dir+'_x'] = df[wind_dir].map(lambda d : np.cos(angle(d)), na_action='ignore')
        df[wind_dir+'_y'] = df[wind_dir].map(lambda d : np.sin(angle(d)), na_action='ignore')
        df = df.drop(wind_dir, axis=1)

    
    meta_q1 = metadata_df.loc['q1',numeric_columns].T
    meta_q3 = metadata_df.loc['q3',numeric_columns].T
    etal_q1 = quantile_df.loc['q1',numeric_columns].T
    etal_q3 = quantile_df.loc['q3',numeric_columns].T

    iou = (meta_q1.combine(etal_q1, min) - meta_q3.combine(etal_q3, max)) / (meta_q1 - meta_q3)
    iou = iou.combine(0, max)
    print(f'Intersection in numeric data:')
    for name in numeric_columns:
        print(f'{name} : {iou[name]}')

    score = 1 - (iou <= 4 / 5).mean()
    
    return df, metadata_df, score
