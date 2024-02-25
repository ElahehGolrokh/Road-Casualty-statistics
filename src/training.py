import importlib
import numpy as np
import pickle


class Trainer:
    """
    Trains ML model from sklearn package

    ...

    Attributes
    ----------
        x_train: input features for training model
        y_train: ground truth labels for training model
        model_path: str path to save trained model
        model: model name from sklearn models, for example LogisticRegression
        classifier_package: classifier package from sklearn for example
                            sklearn.linear_model

    Public Methods
    -------
        train()

    Private Methods
    -------
        _get_model()

    """
    def __init__(self,
                 x_train: np.ndarray,
                 y_train: np.ndarray,
                 model_path: str,
                 model: str,
                 classifier_package: str) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.model_path = model_path
        self.model = model
        self.classifier_package = classifier_package

    def train(self) -> None:
        """Train the model and save it as pickle file"""
        model = self._get_model()  # , model_params: dict
        model.fit(self.x_train, self.y_train)
        with open(self.model_path, 'wb') as file:
            pickle.dump(model, file)

    def _get_model(self, model_params=None):
        """Returns a scikit-learn model."""
        model_class = getattr(importlib.import_module(self.classifier_package),
                              self.model)
        # Instantiates the model
        if model_params:
            model = model_class(**model_params)
        else:
            model = model_class()
        return model
