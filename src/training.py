import importlib
import numpy as np
import pickle
import sklearn

from omegaconf import OmegaConf


class Trainer:
    def __init__(self,
                 config: OmegaConf,
                 x_train: np.ndarray,
                 y_train: np.ndarray,
                 model_path: str,
                 model: str,
                 classifier_package: str) -> None:
        self.config = config
        self.x_train = x_train
        self.y_train = y_train
        self.model_path = model_path
        self.model = model
        self.classifier_package = classifier_package

    def train(self):
        """Train the model and save it as pickle file"""
        model = self._get_model() #, model_params: dict
        model.fit(self.x_train, self.y_train)
        with open(self.model_path, 'wb') as file:
            pickle.dump(model, file)
    
    def _get_model(self, model_params=None) -> sklearn.base.BaseEstimator:
        """Returns a scikit-learn model."""
        model_class = getattr(importlib.import_module(self.classifier_package),
                              self.model)
        if model_params:
            model = model_class(**model_params)  # Instantiates the model
        else:
            model = model_class()
        return model
