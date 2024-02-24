import pandas as pd
from omegaconf.dictconfig import DictConfig
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .utils import read_csv


class DataProvider:
    def __init__(self,
                 config: DictConfig,
                 run_preparation: bool = False,
                 phase: str = None) -> None:
        self.run_preparation = run_preparation
        self.config = config
        self.phase = phase
    
    def run(self):
        if self.run_preparation:
            # run data_preparation
            pass
        else:
            cleaned_data_path = self.config.data.cleaned_data_path
            cleaned_data = read_csv(cleaned_data_path)
            cleaned_data = cleaned_data.sample(frac=1, random_state=325)
        target_col = self.config.features.target
        target = cleaned_data[target_col]
        inputs = cleaned_data.drop(target_col, axis=1)
        scaled_inputs = self._standardize(inputs)
        x, y = self._train_test_split(scaled_inputs, target)
        return x, y
    
    @staticmethod
    def _standardize(inputs: pd.DataFrame):
        
        scalar = StandardScaler()
        scalar.fit(inputs)
        scaled_inputs = scalar.transform(inputs)
        return scaled_inputs
    
    def _train_test_split(self, scaled_inputs, target):

        x_train, x_test, y_train, y_test = train_test_split(scaled_inputs,
                                                            target,
                                                            test_size=self.config.data.test_size,
                                                            random_state=self.config.random_state)
        print(f'x_train.shape = {x_train.shape}',
              f'x_test.shape = {x_test.shape}',
              f'y_train.shape = {y_train.shape}',
              f'y_test.shape = {y_test.shape}')
        if self.phase == 'train':
            return x_train, y_train
        elif self.phase == 'test':
            return x_test, y_test
        else:
            raise ValueError('The passed phase is not defined.')
