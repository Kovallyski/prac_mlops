from numpy import median


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

    metadata_df['type'] = metadata_df['type'].astype(str)
    metadata_df = metadata_df.set_index('type')

    quantile_df['type'] = quantile_df['type'].astype(str)
    quantile_df = quantile_df.set_index('type')

    meta_q1 = metadata_df.loc['q1',numeric_columns].T
    meta_q3 = metadata_df.loc['q3',numeric_columns].T
    etal_q1 = quantile_df.loc['q1',numeric_columns].T
    etal_q3 = quantile_df.loc['q3',numeric_columns].T
    iou = (meta_q1.combine(etal_q1, min) - meta_q3.combine(etal_q3, max)) / (meta_q1 - meta_q3)
    iou = iou.combine(0, max)
    print(f'Intersection in numeric data:')
    for name in numeric_columns:
        print(f'{name} : {iou[name]}')
    if (iou < 0.5).sum() > 1:
        print('DataDrift detected')
    
    score = 1 - (iou < 0.5).mean()
    
    return df, metadata_df, score