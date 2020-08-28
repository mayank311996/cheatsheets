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
data = "./diabetes.csv"

df = pd.read_csv(data)
print(df.shape)
df.head()
df.isnull().sum()
# it looks like null values are written as 0s.

#########################################################################################
# Correlation matrix
corrmat = df.corr()
fig = plt.figure(figsize=(12 ,12))
sns.heatmap(corrmat, vmax=1, square=True, annot=True, vmin=-1)
plt.show()
# no high correlations between features
# so no need to worry about removing correlated features

df.hist(figsize=(12, 12))
plt.show()

sns.pairplot(df, hue='Outcome')
# no separable clusters observed from scatter plots
# that means it will be very hard for machine learning
# model to separate between the class

#########################################################################################
# Feature Engineering
df.Outcome.value_counts()

cols = df.columns
print(cols)

print(f"# rows in dataset {len(df)}")
print("----------------------------")
for col in cols:
    print(
        f"# rows in {len(df.loc[df[col] == 0])} with ZERO value: {col}"
    )

X = df.drop('Outcome', axis=1)
y = df.Outcome

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=10
)

print('Training Set:', len(X_train))
print('Test Set:', len(X_test))
print('Training labels:', len(y_train))
print('Test labels:', len(y_test))

fill = Imputer(missing_values=0, strategy="mean", axis=0)

X_train = fill.fit_transform(X_train)
X_test = fill.fit_transform(X_test)


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
# Model building and evaluation
# Logistic Regression
penalty = ['l1', 'l2']
C = np.logspace(0, 4, 10)
hyperparameters = dict(C=C, penalty=penalty)
fitmodel(X_train, y_train, X_test, y_test, 'Logistic Regression',
         LogisticRegression(), hyperparameters, cv=5)

# XGBoost
param = {
    'n_estimators': [100, 500, 1000, 1500, 2000],
    'max_depth': [2, 3, 4, 5, 6, 7],
    'learning_rate': np.arange(0.01, 0.1, 0.01).tolist()
}
fitmodel(X_train, y_train, X_test, y_test, 'XGBoost',
         XGBClassifier(), param, cv=5)

# Random Forest
param = {
    'n_estimators': [100, 500, 1000, 1500, 2000]
}
fitmodel(X_train, y_train, X_test, y_test, 'Random Forest',
         RandomForestClassifier(), param, cv=5)

# SVC
param = {
    'C': [0.1, 1, 100, 1000],
    'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
}
fitmodel(X_train, y_train, X_test, y_test, 'SVC',
         SVC(), param, cv=5)

#########################################################################################
# Balancing the dataset - over sampling
print(y.value_counts())

sm = SMOTE(random_state=42)
X_res_OS, Y_res_OS = sm.fit_resample(X, y)
pd.Series(Y_res_OS).value_counts()

X_train, X_test, y_train, y_test = train_test_split(
    X_res_OS, Y_res_OS, test_size=0.2, random_state=10
)

fill = Imputer(missing_values=0, strategy="mean", axis=0)

X_train = fill.fit_transform(X_train)
X_test = fill.fit_transform(X_test)

# Now again fit all models and check accuracy and other matrices
#########################################################################################




















