# %matplotlib inline
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
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
from sklearn.manifold import TSNE
from sklearn.preprocessing import Imputer
import pickle
from lightgbm import LGBMClassifier
print('Library Loaded')

#########################################################################################
data_path = './heart.csv'

df = pd.read_csv(data_path)
print(df.shape)
print(df.head())

df.isnull().sum()

cols = df.columns
print(cols)

print(f"# rows in dataset {len(df)}")
print("----------------------------")
for col in cols:
    print(
        f"# rows in {len(df.loc[df[col] == 0])} with ZERO value: {col}"
    )

print(df.dtypes)

#########################################################################################
# visualization
# correlation matrix
corrmat = df.corr()
fig = plt.figure(figsize=(16, 16))
sns.heatmap(corrmat, vmax=1, square=True, annot=True, vmin=-1)
plt.show()
# as seen not much correlation is there between features that means
# we don't need to go into removing highly correlated features or
# feature selection stuff

df.hist(figsize=(12, 12))
plt.show()

sns.barplot(x='sex', y='age', hue='target', data=df)

sns.pairplot(df, hue='target')

# dimensional reduction for visualization
# sometimes we also do this in pipeline like first through PCA
# and then through TSNE
# Or you can use either of them
X = df.drop('target', axis=1)
time_start = time.time()
df_tsne = TSNE(random_state=10).fit_transform(X)
print(f"t-SNE done! Time elapsed: {time.time()-time_start} seconds")

print(df_tsne)


#########################################################################################
def fashion_scatter(x, colors):
    # definition to be added!
    # choose a color palette with seaborn
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette('deep', num_classes))

    # create a scatter plot
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(
        x[:, 0],
        x[:, 1],
        lw=0,
        s=40,
        c=palette[colors.astype(np.int)]
    )
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):
        # position of each label as median of data points
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground='w'),
            PathEffects.Normal()
        ])
        txts.append(txt)

    return f, ax, sc, txts


#########################################################################################
fashion_scatter(df_tsne, df.target)

#########################################################################################
# Feature Engineering
df.target.value_counts()
# not to unbalanced
# we can implement semi-supervised learning techniques later
# to balance the dataset

print(f"# rows in dataset {len(df)}")
print("----------------------------")
for col in cols:
    print(
        f"# rows in {len(df.loc[df[col] == 0])} with ZERO value: {col}"
    )

X = df.drop('target', axis=1)
y = df.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=10
)

print('Training Set:', len(X_train))
print('Test Set:', len(X_test))
print('Training labels:', len(y_train))
print('Test labels:', len(y_test))

# impute with mean all 0 readings as they don't make sense at all
# but what about categorical columns having 1 and 0? they are changed
# as well?
# also you should not fit_transform on test set
# you should just use transform for test set
fill = Imputer(missing_values=0, strategy='mean', axis=0)

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
    'n_estimators': [100, 500, 1000, 1500, 2000],
    'max_depth': [2, 3, 4, 5, 6, 7]
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
# Correcting the mistakes
print(f"# rows in dataset {len(df)}")
print("----------------------------")
for col in cols:
    print(
        f"# rows in {len(df.loc[df[col] == 0])} with ZERO value: {col}"
    )

final_cols = cols
final_cols = list(final_cols)
final_cols.remove('sex')
final_cols.remove('target')
final_cols.remove('age')
print(final_cols)

X = df.drop('target', axis=1)
y = df.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=10
)

# still need to change the imputer stuff
fill = Imputer(missing_values=0, strategy='mean', axis=0)

X_train = fill.fit_transform(X_train)
X_test = fill.fit_transform(X_test)

#########################################################################################

















