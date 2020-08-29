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
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
print("Library Loaded")

#########################################################################################
data = pd.read_csv('creditcard.csv')
print(data.shape)
print(data.head())

print(data.Class.value_counts())

X = data.drop(labels='Class', axis=1)
Y = data['Class']

data.hist(figsize=(20, 20))
plt.show()
# most of the data is already normalized

# correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.show()
# data is not much correlated so we don't need to worry about
# removing correlated features

#########################################################################################
# Feature Engineering
SS = StandardScaler()
X['normAmount'] = SS.fit_transform(X['Amount'].values.reshape(-1, 1))
# because amount column was not normalized
X = X.drop(['Time', 'Amount'], axis=1)
X.head()

np.random.seed(10)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
# we should use stratified split as data is not balanced
print(x_train.shape, x_test.shape)


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


#########################################################################################
# Model Fitting
# Simple Neural Network
model = Sequential([
    Dense(units=16, input_dim=29, activation='relu'),
    Dense(units=23, activation='relu'),
    Dropout(0.5),
    Dense(units=20, activation='relu'),
    Dense(units=24, activation='relu'),
    Dense(units=1, activation='sigmoid')
])
model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(
    x_train,
    y_train,
    batch_size=15,
    epochs=5
)
print(model.evaluate(x_test, y_test))
























