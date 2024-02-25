import numpy as np
import pickle
import uuid

from datetime import datetime
from omegaconf import OmegaConf
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


class Evaluator:
    def __init__(self,
                 config: OmegaConf,
                 x_test: np.ndarray,
                 y_test: np.ndarray,
                 model_path: str,
                 model_name: str) -> None:
        self.config = config
        self.x_test = x_test
        self.y_test = y_test
        self.model_path = model_path
        self.model_name = model_name

    def predict(self):

        with open(self.model_path, 'rb') as model_path:
            model = pickle.load(model_path)
        predictions = model.predict(self.x_test)
        return predictions

    def get_results(self) -> dict:
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
