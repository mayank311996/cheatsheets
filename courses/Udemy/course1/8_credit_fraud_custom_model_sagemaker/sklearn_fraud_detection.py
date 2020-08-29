from __future__ import print_function
import argparse
import os
import pandas as pd
import json
from sklearn import ensemble
from sklearn.externals import joblib


#########################################################################################
def model(args, x_train, y_train, x_test, y_test):
    """
    Trains and outputs Random Forest Classifier
    :param args: input args through arg parser
    :param x_train: training data
    :param y_train: training labels
    :param x_test: testing data
    :param y_test: testing labels
    :return: trained model
    """
    model = ensemble.RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )
    model.fit(x_train, x_test)

    print(f"Training Accuracy: {model.score(x_train, y_train)}")
    print(f"Testing Accuracy: {model.score(x_test, y_test)}")

    return model


def _load_data(file_path, channel):
    """
    Loads the data
    :param file_path: file path to data
    :param channel: channel
    :return: features and labels as data frames
    """
    input_files = [
        os.path.join(file_path, file) for file in os.listdir(file_path)
    ]
    if len(input_files) == 0:
        raise ValueError('There are no files in {}.\n')

    raw_data = [
        pd.read_csv(
            file,
            header=None,
            engine="python"
        ) for file in input_files
    ]
    df = pd.concat(raw_data)

    features = df.iloc[:, 1:].values
    label = df.iloc[:, 0].values

    return features, label


def _parse_args():
    """
    For argument parsing
    :return: Argument parse object with arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n_estimators',
        type=int,
        default=50
    )
    parser.add_argument(
        '--max_depth',
        type=int,
        default=3
    )
    parser.add_argument(
        '--output-data-dir',
        type=str,
        default=os.environ['SM_OUTPUT_DATA_DIR']
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default=os.environ.get('SM_MODEL_DIR')
    )
    parser.add_argument(
        '--train',
        type=str,
        default=os.environ.get('SM_CHANNEL_TRAINING')
    )
    parser.add_argument(
        '--test',
        type=str,
        default=os.environ.get('SM_CHANNEL_TESTING')
    )
    parser.add_argument(
        '--hosts',
        type=list,
        default=json.loads(os.environ.get('SM_HOSTS'))
    )
    parser.add_argument(
        '--current-host',
        type=str,
        default=os.environ.get('SM_CURRENT_HOST')
    )

    return parser.parse_known_args()


def model_fn(model_dir):
    """
    Function to load the trained model
    :param model_dir: Model directory
    :return: Loaded trained classifier
    """
    classifier = joblib.load(
        os.path.join(model_dir, 'model.joblib')
    )
    return classifier


#########################################################################################
if __name__ == '__main__':
    args, unknown = _parse_args()

    train_data, train_labels = _load_data(args.train, 'train')
    eval_data, eval_labels = _load_data(args.test, 'test')

    classifier = model(
        args,
        train_data,
        train_labels,
        eval_data,
        eval_labels
    )

    if args.current_host == args.hosts[0]:
        # Print the coefficients of the trained classifier
        # and save the coefficients
        joblib.dump(
            classifier,
            os.path.join(args.model_dir, 'model.joblib')
        )


#########################################################################################






