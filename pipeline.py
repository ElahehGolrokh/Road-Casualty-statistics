import argparse
import pickle

from dataclasses import dataclass
from omegaconf import OmegaConf
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from src.data_preparation import DataProvider
from src.data_cleaning import DataCleaner


parser = argparse.ArgumentParser(
    prog='pipeline.py',
    description='Scrape Switzerland real state data from' +
                'https://www.homegate.ch/en',
    epilog=f'Thanks for using. Support =>'
)

parser.add_argument('--config',
                    default='config.yaml',
                    help='path to the config file')
parser.add_argument('--output_dir',
                    default='data',
                    help='where to write the prepared data')
parser.add_argument('--clean',
                    action=argparse.BooleanOptionalAction,
                    help='whether to execute the data cleaning step.')
parser.add_argument('--train',
                    action=argparse.BooleanOptionalAction,
                    help='whether to execute the training step.')
parser.add_argument('--evaluate',
                    action=argparse.BooleanOptionalAction,
                    help='whether to execute the evaluation step.')

args = parser.parse_args()


@dataclass
class Pipeline():

    """
    Runs the entire pipline for developing ML model in case that you don't want
    to run each step's scripts one by one.

    ...

    Attributes
    ----------
        config_path: str path to the config file
        output_dir: str where to write the prepared data
        do_clean: bool whether to execute the data cleaning step
        do_train: bool whether to execute the training step
        do_evaluation: bool whether to execute the evaluation step
        run_dir: path to save outputs
        run_info_path: path to info pickle file
        data_dir: path to save prepared data in the root dir
        meta_data_df: pandas dataframe of meta data csv file
        config: DictConf

    Methods
    -------
        start()
        train()
        evaluate()
        export()
        end()
    """

    config_path: str = args.config
    output_dir: str = args.output_dir
    do_clean: bool = args.clean
    do_train: bool = args.train
    do_evaluation: bool = args.evaluate
    run_dir = Path('run')
    # run_info_path = Path('.run.pkl')

    def start(self) -> None:
        """Starting the pipeline"""
        print('Pipeline begins ...')
        config_file_path = Path(self.config_path)
        self.config = OmegaConf.load(config_file_path)
        self.data_dir = self.config.data.data_dir
        print(f'loaded config file {self.config_path}')
        self.run_dir.mkdir(exist_ok=True)

        model_dir = self.run_dir.joinpath('models')
        model_dir.mkdir(exist_ok=True)
        self.model_file = model_dir.joinpath('model')
        if self.do_clean:
            self.clean()
        if self.do_train:
            self.train()
        if self.do_evaluation:
            self.evaluate()
        self.end()
    
    def clean(self) -> None:
        """clean the raw data to be ready for preparation phase"""
        DataCleaner(self.config) # Path(self.output_dir

    def train(self) -> None:
        """Train the model and log to MLFlow and Discord"""

        if self.do_train:
            # print(f'instantiating {self.config.dataset} and config.model ..')
            # dataset = locate(self.config.dataset)(self.config, self.data_dir)
            # model_builder = locate(self.config.model)(self.config)
            # trainer = Trainer(self.config, self.run_dir)

            # tr_gen, n_tr, val_gen, n_val = dataset.create_train_val_generators()

            # with open(self.run_info_path, 'wb') as f:
            #     pickle.dump(dict(self.active_run.info), f)

            data = DataProvider(self.config, phase='train')
            x_train, y_train = data.run()
            model = LogisticRegression()
            model.fit(x_train, y_train)
            with open(self.model_file, 'wb') as file:
                pickle.dump(model, file)

    def evaluate(self) -> None:
        """Evaluate the best model."""

        if self.do_evaluation:
            data = DataProvider(self.config, phase='test')
            x_test, y_test = data.run()

            with open(self.model_file, 'rb') as model_file:
                model = pickle.load(model_file)
            predictions = model.predict(x_test)

            # ## Accuracy
            print('Accuracy = ', accuracy_score(y_test,predictions))
            # ## Precision
            print('Precision = ', precision_score(y_test,predictions))
            # ## Recall
            print('Recall = ', recall_score(y_test,predictions))
            # ## F1 Score
            print('F1_Score = ', f1_score(y_test,predictions))
            # dataset = Dataset(self.config, self.data_dir)

            # checkpoints = get_checkpoints_info(self.run_dir.joinpath('checkpoints'))

    def end(self):
        print('done.')


if __name__ == '__main__':
    Pipeline().start()
