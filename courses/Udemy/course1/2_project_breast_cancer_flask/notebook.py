# %matplotlib inline

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
df_norm = (X - X.mean())/(X.max() - X.min())
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














