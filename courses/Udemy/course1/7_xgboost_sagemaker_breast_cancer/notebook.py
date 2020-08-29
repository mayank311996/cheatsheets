# %matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, \
    classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer
import pickle
from lightgbm import LGBMClassifier
print("Library Loaded")


#########################################################################################
def fitmodel(x_train, y_train, x_test, y_test, algo_name, algorithm, gridsearchparams, cv):
    """
    Trains the model and predicts on test data
    :param x_train: train data
    :param y_train: train labels
    :param x_test: test data
    :param y_test: test labels
    :param algo_name: name of the algorithm
    :param algorithm: algorithm to be used
    :param gridsearchparams: grid parameters
    :param cv: number of cross validations
    :return: none
    """
    np.random.seed(10)

    # Adding below to dump separate csv for train and test for
    # sagemaker
    train = pd.concat([y_train, x_train], axis=1)
    train.to_csv(
        './train.csv',
        index=False,
        header=False,
        columns=columns
    )
    y_train.to_csv(
        './Y-train.csv'
    )
    test = pd.concat([y_test, x_test], axis=1)
    test.to_csv(
        './test.csv',
        index=False,
        header=False,
        columns=columns
    )
    y_test.to_csv(
        './Y-test.csv'
    )

    grid = GridSearchCV(
        estimator=algorithm,
        param_grid=gridsearchparams,
        cv=cv,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )

    grid_result = grid.fit(x_train, y_train)
    best_params = grid_result.best_params_
    pred = grid_result.predict(x_test)
    cm = confusion_matrix(y_test, pred)
    pickle.dump(grid_result, open(algo_name, 'wb'))
    print('Best Params:', best_params)
    print('Classification Report:', classification_report(y_test, pred))
    print('Accuracy Score:' + str(accuracy_score(y_test, pred)))
    print('Confusion Matrix:', cm)

