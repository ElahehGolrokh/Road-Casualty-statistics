import numpy as np
import pickle
import uuid

from datetime import datetime
from sklearn.metrics import f1_score, \
                            accuracy_score, \
                            precision_score, \
                            recall_score


class Evaluator:
    """
    Evaluates a saved ML model

    todo: iMPLEMENT EVALUATION FOR UNLABELED DATA

    ...

    Attributes
    ----------
        x_test: input test features
        y_test: ground truth labels for testing the model
        model_path: str path to saveed trained model
        model_name: str name of the saved model

    Public Methods
    -------
        predict()
        get_results()

    """
    def __init__(self,
                 x_test: np.ndarray,
                 y_test: np.ndarray,
                 model_path: str,
                 model_name: str) -> None:
        self.x_test = x_test
        self.y_test = y_test
        self.model_path = model_path
        self.model_name = model_name

    def predict(self):
        """loads a saved model and returns predictions"""
        with open(self.model_path, 'rb') as model_path:
            model = pickle.load(model_path)
        predictions = model.predict(self.x_test)
        return predictions

    def get_results(self) -> dict:
        """
        Returns a dictionary including unique id, saved model_name,
        evaluation_date and the following sklearn metrics:
        accuracy
        precision
        recall
        f1_score
        """
        predictions = self.predict()
        report_dict = {
            'id': str(uuid.uuid1()),
            'model_name': self.model_name,
            'evaluation_date': str(datetime.now()),
            'accuracy': accuracy_score(self.y_test, predictions),
            'precision': precision_score(self.y_test, predictions),
            'recall': recall_score(self.y_test, predictions),
            'f1_score': f1_score(self.y_test, predictions)
        }
        return report_dict
