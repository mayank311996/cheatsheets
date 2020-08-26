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























