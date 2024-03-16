import numpy as np
import os
import warnings
import zipfile

from omegaconf.dictconfig import DictConfig

from .utils import read_csv

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
import pandas as pd


class DataCleaner:
    """
    Cleans the raw version of the data in this link:
    https://www.kaggle.com/datasets/juhibhojani/road-accidents-data-2022/data

    ...

    Attributes
    ----------
        config: DictConf of config file
        data_dir: str path to data directory

    Private Methods
    -------
        _return_path()
        _unzip_data()
        _read_raw_csv()
        _remove_non_informative()
        _remove_outliers()
        _dealing_with_nans()
        _dealing_with_categories()
        _get_dummies()

    Public Methods
    -------
        clean_df()

    """
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.data_dir = self.config.data.data_dir

    def _return_path(self) -> str:
        """
        Returns path to zipped file stored in the data directory
        """
        raw_data_path = self.config.data.raw_data_path
        try:
            dir_path = os.path.join(self.data_dir, raw_data_path)
        except FileNotFoundError:
            print('There are some problems about your file paths.')
        else:
            return dir_path

    def _unzip_data(self) -> None:
        """
        Unzips zipped csv file in the data_dir which is defined
        in cinfig file
        """
        dir_path = self._return_path()
        with zipfile.ZipFile(dir_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)

    def _read_raw_csv(self) -> pd.DataFrame:
        """
        Gets path to raw csv file which is stored in data directory
        """
        self._unzip_data()
        csv_file = next(os.walk(self.data_dir))[2][1]
        file_path = os.path.join(self.data_dir, csv_file)
        df = read_csv(file_path)
        return df

    def _remove_non_informative(self) -> pd.DataFrame:
        """
        Removes non-informative columns which are defined in
        the config file
        """
        df = self._read_raw_csv()
        non_informatives = self.config.features.non_informatives
        df.drop(non_informatives, axis=1, inplace=True)
        return df

    def _remove_outliers(self, perc: float = .99) -> pd.DataFrame:
        """Removes outliers from numerical columns"""
        df = self._remove_non_informative()
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

    def _dealing_with_nans(self) -> pd.DataFrame:
        """Replace nan values with np.nan"""
        df = self._remove_outliers()
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

    def _dealing_with_categories(self) -> pd.DataFrame:
        """
        Names different categories to ba able to analyze better
        """
        df = self._dealing_with_nans()

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
        """Converts categorical columns with pandas get_dummies method"""
        df = self._dealing_with_categories()
        df = pd.get_dummies(df, drop_first=True)
        return df

    def clean_df(self, data_dir: str) -> None:
        """Saves the cleaned data"""
        df = self._get_dummies()
        file_path = os.path.join(data_dir, 'data_cleaned.csv')
        df.to_csv(file_path, index=False)


class InferenceCleaner:
    """
    Cleans the input for inference phase

    ...

    Attributes
    ----------
        config: DictConf of config file
        data_path: str path to test data

    Private Methods
    -------
        _read_raw_csv()
        _remove_non_informative()
        _is_nans()

    Public Methods
    -------
        create_final_df()

    """
    def __init__(self, config: DictConfig, data_path: str) -> None:
        self.config = config
        self.data_path = data_path

    def _read_raw_csv(self) -> pd.DataFrame:
        """
        Returns pandas dataframe of input csv file
        """
        df = read_csv(self.data_path)
        return df

    def _remove_non_informative(self) -> pd.DataFrame:
        """
        Removes non-informative columns which are defined in
        the config file
        """
        df = self._read_raw_csv()
        non_informatives = self.config.features.non_informatives
        df.drop(non_informatives, axis=1, inplace=True)
        return df

    def _is_nans(self) -> pd.DataFrame:
        """Checks if there are any nan values in input data"""
        df = self._remove_non_informative()
        nan_1 = self.config.features.nan_1
        nan_1_9 = self.config.features.nan_1_9

        for col in nan_1:
            df[col][df[col] == -1] = np.nan

        for col in nan_1_9:
            df[col][(df[col] == -1) | (df[col] == 9)] = np.nan

        if df.isnull().sum().any():
            raise ValueError('There are some nan values in your input data')
        return df

    def create_final_df(self) -> pd.DataFrame:
        """
        Modifies the columns of the cleaned test data in a way that the
        final df would be the same as the trained df which has been used
        for training the saved model.
        """
        df = self._is_nans()
        df['casualty_class_Passenger'] = self.create_columns(df,
                                                             'casualty_class',
                                                             'Passenger')
        df['casualty_class_Pedestrian'] = self.create_columns(df,
                                                              'casualty_class',
                                                              'Pedestrian')
        df['sex_of_casualty_Male'] = self.create_columns(df,
                                                         'sex_of_casualty',
                                                         'Male')
        df['Car'] = self.create_columns(df,
                                        'casualty_type',
                                        'Car')
        df['Cyclist'] = self.create_columns(df,
                                            'casualty_type',
                                            'Cyclist')
        df['Motorcycle'] = self.create_columns(df,
                                               'casualty_type',
                                               'Motorcycle')
        df['Other'] = self.create_columns(df,
                                          'casualty_type',
                                          'Other_vehicle_occupant')
        df['Pedestrian'] = self.create_columns(df,
                                               'casualty_type',
                                               'Pedestrian')
        df['Van'] = self.create_columns(df,
                                        'casualty_type',
                                        'Van')
        df['Urban_area'] = self.create_columns(df,
                                               'casualty_home_area_type',
                                               'Urban_area')
        df['middle_deprived'] = self.create_columns(df,
                                                    'casualty_imd_decile',
                                                    'middle_deprived')
        df['more_deprived'] = self.create_columns(df,
                                                  'casualty_imd_decile',
                                                  'more_deprived')
        df.drop(['casualty_class', 'sex_of_casualty', 'casualty_type',
                 'casualty_home_area_type', 'casualty_imd_decile'],
                axis=1,
                inplace=True)
        return df

    @staticmethod
    def create_columns(df, old_col, value):
        """Fills new values based on old column values"""
        return df[old_col].apply(lambda x: 1 if x == value else 0)
