import argparse

from dataclasses import dataclass
from omegaconf import OmegaConf
from pathlib import Path

from sample_settings import SUPPORT
from src.data_preparation import DataProvider
from src.data_cleaning import DataCleaner
from src.evaluation import Evaluator
from src.managing_report import ReportManager
from src.training import Trainer


parser = argparse.ArgumentParser(
    prog='pipeline.py',
    description='running pipeline for developing ML model' +
                'for analysing road casualty statistics',
    epilog=f'Thanks for using. => {SUPPORT["email"]} '
)

parser.add_argument('--config',
                    default='config.yaml',
                    help='path to the config file')
parser.add_argument('--model_name',
                    default='model',
                    help='The name used for saving the trained model')
parser.add_argument('--clean',
                    action=argparse.BooleanOptionalAction,
                    help='whether to execute the data cleaning step.')
parser.add_argument('--train',
                    action=argparse.BooleanOptionalAction,
                    help='whether to execute the training step.')
parser.add_argument('--model',
                    default='LogisticRegression',
                    help='choose one of the sklearn classification models')
parser.add_argument('--classifier_package',
                    default='sklearn.linear_model',
                    help='choose one of the sklearn classification models')
parser.add_argument('--evaluate',
                    action=argparse.BooleanOptionalAction,
                    help='whether to execute the evaluation step.')

args = parser.parse_args()


@dataclass
class Pipeline():

    """
    Runs the entire pipline for developing ML model

    ...

    Attributes
    ----------
        config_path: str path to the config file
        do_clean: bool whether to execute the data cleaning step
        do_train: bool whether to execute the training step
        do_evaluation: bool whether to execute the evaluation step
        run_dir: path to save reports and models directory
        data_dir: path to data dir in which the cleaned data is stored
                  as csv file
        config: DictConf config file
        model_name: str name for saving the trained model
        model: model name from sklearn models, for example LogisticRegression
        classifier_package: classifier package from sklearn for example
                            sklearn.linear_model
        model_path: path for saving the trained model

    Public Methods
    -------
        start()
        clean()
        train()
        evaluate()
        end()
    """

    config_path: str = args.config
    do_clean: bool = args.clean
    do_train: bool = args.train
    model_name: str = args.model_name
    model: str = args.model
    classifier_package: str = args.classifier_package
    do_evaluation: bool = args.evaluate
    run_dir = Path('run')

    def start(self) -> None:
        """Starting the pipeline"""
        print('Pipeline begins ...')
        # reading config file
        config_file_path = Path(self.config_path)
        self.config = OmegaConf.load(config_file_path)
        self.data_dir = self.config.data.data_dir
        print(f'loaded config file {self.config_path}')

        # creating run directory
        self.run_dir.mkdir(exist_ok=True)

        # dir and name for saving model
        model_dir = self.run_dir.joinpath('models')
        model_dir.mkdir(exist_ok=True)
        self.model_path = model_dir.joinpath(self.model_name)

        # run pipeline
        if self.do_clean:
            self.clean()
        if self.do_train:
            self.train()
        if self.do_evaluation:
            self.evaluate()
        self.end()

    def clean(self) -> None:
        """clean the raw data to be ready for preparation phase"""
        DataCleaner(self.config).clean_df(self.data_dir)

    def train(self) -> None:
        """Train the model and save it as pickle file"""
        if self.do_train:
            data = DataProvider(self.config, phase='train')
            x_train, y_train = data.run()

            trainer = Trainer(x_train=x_train,
                              y_train=y_train,
                              model_path=self.model_path,
                              model=self.model,
                              classifier_package=self.classifier_package)
            trainer.train()

    def evaluate(self) -> None:
        """Evaluate the best model."""

        if self.do_evaluation:
            data = DataProvider(self.config, phase='test')
            x_test, y_test = data.run()

            # evaluate the model named: model_name
            evaluator = Evaluator(x_test=x_test,
                                  y_test=y_test,
                                  model_path=self.model_path,
                                  model_name=self.model_name)
            report_dict = evaluator.get_results()

            # update report file
            report_manager = ReportManager()
            report_manager.update_json(report_dict)

    def end(self):
        print('done.')


if __name__ == '__main__':
    Pipeline().start()
