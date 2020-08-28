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

cols = df.columns
print(cols)

print(f"# rows in dataset {len(df)}")
print("----------------------------")
for col in cols:
    print(
        f"# rows in {len(df.loc[df[col] == 0])} with ZERO value: {col}"
    )





















