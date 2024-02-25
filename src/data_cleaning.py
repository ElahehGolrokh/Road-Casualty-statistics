import numpy as np
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
import pandas as pd
import zipfile

from .utils import read_csv


class DataCleaner:
    def __init__(self, config) -> None:
        self.config = config
        self.data_dir = self.config.data.data_dir
    
    def return_path(self):
        raw_data_path = self.config.data.raw_data_path
        try:
            dir_path = os.path.join(self.data_dir, raw_data_path)
        except FileNotFoundError:
            print('There are some problems about your file paths.')
        else:
            return dir_path
    
    def unzip_data(self):
        """
        Unzips zipped csv file in the data_dir which is defined
        in cinfig file
        """
        dir_path = self.return_path()
        with zipfile.ZipFile(dir_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
    
    def read_raw_csv(self):
        self.unzip_data()
        csv_file = next(os.walk(self.data_dir))[2][1]
        file_path = os.path.join(self.data_dir, csv_file)
        df = read_csv(file_path)
        return df
    
    def remove_non_informative(self):
        df = self.read_raw_csv()
        non_informatives = self.config.features.non_informatives
        df.drop(non_informatives, axis=1, inplace=True)
        return df
    
    def remove_outliers(self, perc: float=.99):
        # Remove outliers
        df = self.remove_non_informative()
        numerical_features = self.config.features.numerical_features
        for col in numerical_features:
            if 'age' in col:
                q = df[col].quantile(perc)
                df = df[df[col] < q]
            else:
                for i in df[col].index:
                    if df[col].loc[i] > 60:
                        df.drop(i, inplace=True)
        return df
    
    def dealing_with_nans(self):
        # Replace nan values with np.nan
        df = self.remove_outliers()
        nan_1 = self.config.features.nan_1
        nan_1_9 = self.config.features.nan_1_9

        for col in nan_1:
            df[col][df[col] == -1] = np.nan

        for col in nan_1_9:
            df[col][(df[col] == -1) | (df[col] == 9)] = np.nan

        df.dropna(inplace=True)
        # set index
        df = df.set_index(np.arange(len(df)))
        return df
    
    def _dealing_with_categories(self):
        # DEAL WITH CATEGORICALS

        # name different categories to ba able to analyze better
        df = self.dealing_with_nans()

        col = 'casualty_imd_decile'
        df[col][(df[col] == 2) |
                (df[col] == 3) |
                (df[col] == 1)] = 'more_deprived'
        df[col][(df[col] == 4) |
                (df[col] == 5) |
                (df[col] == 6) |
                (df[col] == 7)] = 'middle_deprived'
        df[col][(df[col] == 8) |
                (df[col] == 9) |
                (df[col] == 10)] = 'less_deprived'

        col = 'casualty_home_area_type'
        df[col][df[col] == 1] = 'Urban_area'
        df[col][(df[col] == 3) |
                (df[col] == 2)] = 'Rural or town'

        col = 'casualty_type'
        df[col][df[col] == 0] = 'Pedestrian'
        df[col][df[col] == 1] = 'Cyclist'
        df[col][(df[col] == 2) |
                (df[col] == 3) |
                (df[col] == 4) |
                (df[col] == 5) |
                (df[col] == 23) |
                (df[col] == 97)] = 'Motorcycle'
        df[col][(df[col] == 8) |
                (df[col] == 9)] = 'Car'
        # df[col][df[col] == 10] = 'Minibus'
        df[col][(df[col] == 10) |
                (df[col] == 11)] = 'Bus or Minibus'
        df[col][df[col] == 16] = 'Horse_rider'
        df[col][df[col] == 17] = 'Agricultural_vehicle'
        df[col][df[col] == 18] = 'Tram_occupant'
        df[col][(df[col] == 19) |
                (df[col] == 20) |
                (df[col] == 21) |
                (df[col] == 98)] = 'Van'
        df[col][df[col] == 90] = 'Other_vehicle_occupant'
        df[col][df[col] == 22] = 'Scooter'

        df.drop(df[(df['casualty_type'] == 'Horse_rider') |
                   (df['casualty_type'] == 'Agricultural_vehicle') |
                   (df['casualty_type'] == 'Tram_occupant') |
                   (df['casualty_type'] == 'Scooter')].index,
                inplace=True)

        col = 'casualty_severity'
        df[col][(df[col] == 2) |
                (df[col] == 1)] = 'Serious'
        df[col][df[col] == 3] = 'Slight'

        col = 'sex_of_casualty'
        df[col][df[col] == 1] = 'Male'
        df[col][df[col] == 2] = 'Female'

        col = 'casualty_class'
        df[col][df[col] == 1] = 'Driver or rider'
        df[col][df[col] == 2] = 'Passenger'
        df[col][df[col] == 3] = 'Pedestrian'
        return df
    
    def _get_dummies(self) -> pd.DataFrame:
        df = self._dealing_with_categories()
        df = pd.get_dummies(df, drop_first=True)
        return df
    
    def clean_df(self, data_dir: str) -> None:
        df = self._get_dummies()
        df = df.sample(frac=1, random_state=325)
        file_path = os.path.join(data_dir, 'data_cleaned.csv')
        df.to_csv(file_path, index=False)
