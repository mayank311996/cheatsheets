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


























