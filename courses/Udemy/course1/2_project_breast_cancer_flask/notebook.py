# %matplotlib inline

import pickle
import numpy as np
import pandas as pd
import seaborn as s
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_Selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, \
    accuracy_score
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from skelarn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

#########################################################################################
data = pd.read_csv('/input/data.csv')

print(data.shape)
data.head()

df = data.drop('Unnamed: 32', axis=1)
df['diagnosis'].value_counts()

df.diagnosis = df.diagnosis.astype('category')
df.head()

X = df.drop(labels='diagnosis', axis=1)
Y = df['diagnosis']
col = X.columns
print(col)

#########################################################################################
# Feature Engineering
print(X.isnull().sum())

# Normalization and Feature Scaling
df_norm = (X - X.mean()) / (X.max() - X.min())
df_norm = pd.concat([df_norm, Y], axis=1)
df_norm.head()

# Visualization (To be done on original data not scaled data)
plt.rcParams['figure.figsize'] = (12, 8)
s.set(font_scale=1.4)
s.heatmap(df.drop('diagnosis', axis=1).drop('id', axis=1).corr(),
          cmap='coolwarm')  # why to remove diagnosis column?

plt.rcParams['figure.figsize'] = (10, 5)
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
s.boxplot('diagnosis', y='radius_mean', data=df, ax=ax1)
s.boxplot('diagnosis', y='texture_mean', data=df, ax=ax2)
s.boxplot('diagnosis', y='perimeter_mean', data=df, ax=ax3)
s.boxplot('diagnosis', y='area_mean', data=df, ax=ax4)
s.boxplot('diagnosis', y='smoothness_mean', data=df, ax=ax5)
f.tight_layout()

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
s.boxplot('diagnosis', y='compactness_mean', data=df, ax=ax1)
s.boxplot('diagnosis', y='concavity_mean', data=df, ax=ax2)
s.boxplot('diagnosis', y='concave_points_mean', data=df, ax=ax3)
s.boxplot('diagnosis', y='symmetry_mean', data=df, ax=ax4)
s.boxplot('diagnosis', y='fractal_dimension_mean', data=df, ax=ax5)
f.tight_layout()
# in this case we won't remove outliers as dataset is small.
# Also, for specific domain like medical, finance etc.
# outliers can represent very valuable information

# To get idea that data is normally distributed or gaussian
# distributed or skewed etc.
g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')
g.map(s.distplot, 'radius_mean', hist=False, rug=True)

g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')
g.map(s.distplot, 'texture_mean', hist=True, rug=True)

g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')
g.map(s.distplot, 'perimeter_mean', hist=True, rug=True)

g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')
g.map(s.distplot, 'area_mean', hist=True, rug=True)

g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')
g.map(s.distplot, 'smoothness_mean', hist=True, rug=True)

g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')
g.map(s.distplot, 'compactness_mean', hist=True, rug=True)

g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')
g.map(s.distplot, 'concavity_mean', hist=True, rug=True)

g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')
g.map(s.distplot, 'concave_points_mean', hist=True, rug=True)

g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')
g.map(s.distplot, 'symmetry_mean', hist=True, rug=True)

g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')
g.map(s.distplot, 'fractal_dimension_mean', hist=True, rug=True)
# over here we are not using any normalization techniques as
# data looks normalized (Bell curve)

plt.rcParams['figure.figsize'] = (10, 5)
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
s.boxplot('diagnosis', y='radius_se', data=df, ax=ax1,
          palette='cubehelix')
s.boxplot('diagnosis', y='texture_se', data=df, ax=ax2,
          palette='cubehelix')
s.boxplot('diagnosis', y='perimeter_se', data=df, ax=ax3,
          palette='cubehelix')
s.boxplot('diagnosis', y='area_se', data=df, ax=ax4,
          palette='cubehelix')
s.boxplot('diagnosis', y='smoothness_se', data=df, ax=ax5,
          palette='cubehelix')
f.tight_layout()

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
s.boxplot('diagnosis', y='compactness_se', data=df, ax=ax1,
          palette='cubehelix')
s.boxplot('diagnosis', y='concavity_se', data=df, ax=ax2,
          palette='cubehelix')
s.boxplot('diagnosis', y='concave_points_se', data=df, ax=ax3,
          palette='cubehelix')
s.boxplot('diagnosis', y='symmetry_se', data=df, ax=ax4,
          palette='cubehelix')
s.boxplot('diagnosis', y='fractal_dimension_se', data=df, ax=ax5,
          palette='cubehelix')
f.tight_layout()

plt.rcParams['figure.figsize'] = (10, 5)
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
s.boxplot('diagnosis', y='radius_worst', data=df, ax=ax1,
          palette='coolwarm')
s.boxplot('diagnosis', y='texture_worst', data=df, ax=ax2,
          palette='coolwarm')
s.boxplot('diagnosis', y='perimeter_worst', data=df, ax=ax3,
          palette='coolwarm')
s.boxplot('diagnosis', y='area_worst', data=df, ax=ax4,
          palette='coolwarm')
s.boxplot('diagnosis', y='smoothness_worst', data=df, ax=ax5,
          palette='coolwarm')
f.tight_layout()

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
s.boxplot('diagnosis', y='compactness_worst', data=df, ax=ax1,
          palette='coolwarm')
s.boxplot('diagnosis', y='concavity_worst', data=df, ax=ax2,
          palette='coolwarm')
s.boxplot('diagnosis', y='concave_points_worst', data=df, ax=ax3,
          palette='coolwarm')
s.boxplot('diagnosis', y='symmetry_worst', data=df, ax=ax4,
          palette='coolwarm')
s.boxplot('diagnosis', y='fractal_dimension_worst', data=df, ax=ax5,
          palette='coolwarm')
f.tight_layout()
# highly correlated data can be remove as they don't add much value
# also removing them saves computation time.
# this can be useful when you have large number of features say 300

X_norm = df_norm.drop(labels='diagnosis', axis=1)
Y_norm = df_norm['diagnosis']
col = X_norm.columns

le = LabelEncoder()
le.fit(Y_norm)

Y_norm = le.transform(Y_norm)
Y_norm = pd.DataFrame(Y_norm)
Y_norm.head()


#########################################################################################
# Fitting the ML model


def fitmodel(x, y, algo_name, algorithm, gridsearchparams, cv):
    """
    Trains the model and predicts on test data
    :param x: data
    :param y: labels
    :param algo_name: name of the algorithm
    :param algorithm: algorithm to be used
    :param gridsearchparams: grid parameters
    :param cv: number of cross validations
    :return: none
    """
    np.random.seed(10)
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.2)

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


# SVC ML Model
param = {
    'C': [0.1, 1, 100, 1000],
    'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
}
fitmodel(X_norm, Y_norm, 'SVC', SVC(), param, cv=5)

# Random Forest
param = {
    'n_estimators': [100, 500, 1000, 2000]
}
fitmodel(X, Y, 'Random Forest', RandomForestClassifier(), param, cv=10)
# not used normalized data (remember Abhishek Thakur book tree based \
# algorithms don't need any scaling)

# repeated
np.random.seed(10)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

forest = RandomForestClassifier(n_estimators=1000)
fit = forest.fit(x_train, y_train)
accuracy = fit.score(x_test, y_test)
predict = fit.predict(x_test)
cmatrix = confusion_matrix(y_test, predict)

print(f'Accuracy of Random Forest: {accuracy}')

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

print('Feature ranking:')
for f in range(X.shape[1]):
    print(f"feature {list(X)[f]} ({importances[indices[f]]})")

feat_imp = pd.DataFrame({
    'Feature': list(X),
    'Gini importance': importances[indices]
})
plt.rcParams['figure.figsize'] = (12, 12)
s.set_style('whitegrid')
ax = s.barplot(x='Gini importance', y='Feature', data=feat_imp)
ax.set(xlabel='Gini Importance')
plt.show()

# XGBoost
param = {
    'n_estimators': [100, 500, 1000, 2000]
}
fitmodel(X, Y, 'XGBoost', XGBClassifier(), param, cv=5)

# Balancing the Dataset
df_norm.head()

sm = SMOTE(random_state=42)
X_res, Y_res = sm.fit_resample(X_norm, Y_norm)

pd.Series(Y_res).value_counts()

param = {
    'n_estimators': [100, 500, 1000, 2000]
}
fitmodel(X_res, Y_res, 'Random Forest', RandomForestClassifier(),
         param, cv=10)

param = {
    'C': [0.1, 1, 100, 1000],
    'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
}
fitmodel(X_res, Y_res, 'SVC', SVC(), param, cv=5)

# Feature Selection
feat_imp.index = feat_imp.Feature
feat_to_keep = feat_imp.ilov[1:15].index
print(feat_to_keep)

X_res = pd.DataFrame(X_res)
Y_res = pd.DataFrame(Y_res)
X_res.columns = X_norm.columns

param = {
    'n_estimators': [100, 500, 1000, 2000]
}
fitmodel(X_res[feat_to_keep], Y_res, 'Random Forest',
         RandomForestClassifier(), param, cv=10)

# Reloading the saved model
loaded_model = pickle.load(open("XGBoost_norm", "rb"))

pred1 = loaded_model.predict(x_test)
print(loaded_model.best_params_)

#########################################################################################








