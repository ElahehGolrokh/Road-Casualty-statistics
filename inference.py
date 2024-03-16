import argparse
import pickle

from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler

from src.data_cleaning import InferenceCleaner
from sample_settings import SUPPORT


parser = argparse.ArgumentParser(
    prog='inference.py',
    description='running inference for getting predictions based on' +
                'developed ML model for road casualty statistics',
    epilog=f'Thanks for using. => {SUPPORT["email"]} '
)

parser.add_argument('--config_path',
                    default='config.yaml',
                    help='path to config file')
parser.add_argument('--model_path',
                    default='run/models/model',
                    help='path to saved ML model')
parser.add_argument('--data_path',
                    default='data/test.csv',
                    help='path to test csv file')
args = parser.parse_args()


class Inference:
    def __init__(self,
                 model_path: str = args.model_path,
                 data_path: str = args.data_path,
                 config_path: str = args.config_path) -> None:
        self.model_path = model_path
        self.data_path = data_path
        self.config = OmegaConf.load(config_path)

    def load_model(self):
        """loads a saved model"""
        with open(self.model_path, 'rb') as model_path:
            model = pickle.load(model_path)
        return model

    def preprocess_inputs(self):
        """Preprocess and standardize the input features"""
        preprocesses_inputs = InferenceCleaner(self.config,
                                               self.data_path).create_final_df()
        scalar = StandardScaler()
        scalar.fit(preprocesses_inputs)
        scaled_inputs = scalar.transform(preprocesses_inputs)
        return scaled_inputs

    def predict(self) -> list:
        """Returns model predictions"""
        model = self.load_model()
        x_test = self.preprocess_inputs()
        predictions = model.predict(x_test)
        return predictions

    def print_results(self):
        """Prints the model predictions"""
        predictions = self.predict()
        prediction = ['Slight' if pred else 'Serious' for pred in predictions]
        print('prediction = ', prediction)


if __name__ == '__main__':
    Inference().print_results()
