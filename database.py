import os
import pathlib
import time
import pandas as pd

from utils import display

class DataBase:
    category_columns = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
    target_columns = ['RainTomorrow']
    numeric_columns = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
        'Sunshine', 'WindGustSpeed',
        'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
        'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
        'Temp3pm']
    datetime_columns = ['Date']
    index_columns = ['Date', 'Location']
    all_columns = index_columns+category_columns+numeric_columns

    def __init__(self, conf):
        self.path = conf['database_path']
        self.path = pathlib.Path(self.path)
        if not os.path.exists(self.path):
            self.df = pd.DataFrame(columns=self.all_columns+self.target_columns+['batchid', 'known'])
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.df.to_csv(self.path, index=False)
        else:
            self.df = pd.read_csv(self.path)
        
        self.metadata_path = conf['metadata_path']
        self.metadata_path = pathlib.Path(self.metadata_path)
        if not os.path.exists(self.metadata_path):
            self.metadata_df = pd.DataFrame(columns=self.numeric_columns+self.category_columns+['batchid','type'])
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.metadata_df.to_csv(self.metadata_path, index=False)
        else:
            self.metadata_df = pd.read_csv(self.metadata_path)

        
        self.quantile_path = conf['quantile_path']
        self.quantile_path = pathlib.Path(self.quantile_path)
        if not os.path.exists(self.quantile_path):
            self.quantile_df = pd.DataFrame(columns=self.numeric_columns+self.category_columns+['type'])
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.quantile_df.to_csv(self.quantile_path, index=False)
        else:
            self.quantile_df = pd.read_csv(self.quantile_path)
            
    def __get_metadata(self, df):
        shape = df.shape
        full_duplicated = df.duplicated()
        index_duplicated = df[self.index_columns].duplicated()
        collisions = (~ df[index_duplicated].duplicated())

        print(f'Number of duplicated datas: {full_duplicated.sum()}')
        print(f'among them duplicated by date and location: {index_duplicated.sum()}')
        print(f'Number of date&location collisions: {collisions.sum()}')
        nacompletness = df[self.numeric_columns + self.category_columns].isna()
        print(f'Missing values: {nacompletness.sum().sum()} ({100*nacompletness.sum().sum()/shape[0]:}%)')
        print(f'among them:')
        for name in self.numeric_columns+self.category_columns:
            p = df[name].isna().sum()
            if p > 0:
                print(f'{name} : {p} ({p/shape[0]:}%)')

        metadata_df = pd.DataFrame([df[self.numeric_columns].max(),df[self.numeric_columns].min(),df[self.numeric_columns].quantile(0.25),df[self.numeric_columns].quantile(0.75),nacompletness[self.numeric_columns+self.category_columns].sum(),nacompletness[self.numeric_columns+self.category_columns].sum()/shape[0]])
        metadata_df['type'] = ['max', 'min', 'q3', 'q1', 'na', 'na%']

        return metadata_df

    def __load_data(self, path):
        if not os.path.exists(path):
            print(f'File {path} does not exist')
            return None
        df = pd.read_csv(path)
        df = df.reindex(sorted(self.df.columns), axis=1)


        #CHECK ALL COLUMNS
        all_columns = self.all_columns
        not_found = pd.Index(all_columns).difference(df.columns.intersection(all_columns))
        display(not_found)
        if len(not_found):
            print(f'Columns {not_found} not found in new data')
            return None

        metadata_df = self.__get_metadata(df)
        metadata_df = metadata_df.reindex(sorted(self.metadata_df.columns), axis=1)

        return df, metadata_df

    
    def load_test(self, path):
        df, metadata_df = self.__load_data(path)

        return df, metadata_df
    
    def load_train(self, path):
        df, metadata_df = self.__load_data(path)

        not_found = pd.Index(DataBase.target_columns).difference(df.columns.intersection(DataBase.target_columns))
        if len(not_found):
            print(f'Target columns {not_found} not found in new data')
            return None

        batchid = time.time()
        print(f'batchid : {batchid}')

        df['known'] = False
        df['batchid'] = batchid
        metadata_df['batchid'] = batchid

        df = df[self.all_columns+self.target_columns+['batchid', 'known']]
        df.to_csv(self.path, index=False, mode='a', header=False)

        metadata_df.to_csv(self.metadata_path, index=False, mode='a', header=False)

        self.df = pd.concat([self.df,df])
        self.df['known'] = self.df['known'].astype(bool)
        self.metadata_df = pd.concat([self.metadata_df, metadata_df])

        return batchid
    
    def get_data(self, batchids=None):
        df = []
        metadata_df = []

        if batchids is not None:
            for batchid in batchids:
                df.append(self.df[self.df['batchid']==batchid].copy())
                metadata_df.append(self.metadata_df[self.metadata_df['batchid']==batchid].copy())
            df = pd.concat(df)
            metadata_df = pd.concat(metadata_df)
        else:
            df = self.df.copy()
            metadata_df = self.metadata_df.copy()


        cur_df = df.drop(['batchid', 'known'], axis=1)
        return cur_df, self.__get_metadata(cur_df), metadata_df.drop('batchid', axis=1), self.quantile_df.copy()

    def get_unknown(self):
        return sorted(set(self.df[~self.df['known']]['batchid']))
    
    def get_known(self):
        return sorted(set(self.df[self.df['known']]['batchid']))

    def set_known(self, batchids=None):
        if batchids is not None:
            for batchid in batchids:
                self.df[self.df['batchid']==batchid]['known'] = True
        else:
            self.df['known'] = True


        self.quantile_df = self.__get_metadata(self.df.drop(['batchid', 'known'], axis=1))
        self.quantile_df.to_csv(self.quantile_path, index=False)
        self.df.to_csv(self.path, index=False)