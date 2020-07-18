# Approaching Almost Any Machine Learning Problem

#################################################################################################
# Chapter 1: Seeting Up Your Working Environment
#################################################################################################

$ cd ~/Downloads
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ sh Miniconda3-latest-Linux-x86_64.sh
$ conda create -n environment_name python=3.7.6
$ conda activate environment_name

### To create environment from yml file

$ conda env create -f environment.yml
$ conda activate ml


#################################################################################################
# Chapter 2: Supervised vs Unsupervised learning 
#################################################################################################

### t-SNE visuallization of MNIST dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import datasets 
from sklearn import manifold

% matplotlib inline 

data = datasets.fetch_openml(
    'mnist_784',
    version=1,
    return_X_y=True
)

pixel_values, targets = data
targets = targets.astype(int) # Important in order to save space and memory. Also, bydefault targets were string type so need to convert to int.

# To visualize a single image
single_image = pixed_values[1, :].reshape(28,28)
plt.imshow(single_image, cmap='gray')

tsne = manifold.TSNE(n_components=2, random_state=42)
transformed_data = tsne.fit_transform(pixel_values[:3000, :])

tsne_df = pd.DataFrame(
    np.column_stack((transformed_data, targets[:3000])),
    columns=["x","y","targets"]
)

tsne_df.loc[:,"targets"] = tsne_df.targets.astype(int)

grid = sns.FacetGrid(tsne_df, hue="targets", size=8)
grid.map(plt.scatter, "x", "y").add_legend()



#################################################################################################
# Chapter 3: Cross-validation
#################################################################################################

### Explaining Overfitting
# Using wine quality dataset
import pandas as pd
df = pd.read_csv("winequality-red.csv")

# a mapping dictionary that maps the quality values from 0 to 5
quality_mapping = {
    3: 0,
    4: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5
}

# we can use the map function of pandas with any dictionary to convert the values in a given column to values in the dictionery
df.loc[:, "quality"] = df.quality.map(quality_mapping)

# Splitting data into training and testing 
# Use sample with frac=1 to shuffle the dataframe. We reset the indices since they change after shuffling the data frame.
df = df.sample(frac=1).reset_index(drop=True)

df_train = df.head(1000)
df_test = df.tail(599)

from sklearn import tree
from sklearn import metrics 

clf = tree.DecisionTreeClassifier(max_depth=3)

# Choosing some cloumns to train the model

cols = [
    'fixed acidity', 
    'volatile acidity', 
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol' 
]

clf.fit(df_train[cols], df_train.quality)

train_predictions = clf.predict(df_train[cols])
test_predictions = clf.predict(df_test[cols])

train_accuracy = metrics.accuracy_score(
    df_train.quality, train_predictions
)
test_accuracy = metrics.accuracy_score(
    df_test.quality, test_predictions
)

### Getting accuracy vs tree depth plot 
from sklearn import tree
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# This is our global size of label text on the plots
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)

% matplotlib inline

# Initialize lists to store accuracies for training and testing data, we start with 50% accuracy
train_accuracies = [0.5]
test_accuracies = [0.5]

# Iterating over depth
for depth in range(1,25):
    # init the model
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    # cols for training (this can be done outside the loop and should be done that way)
    cols = [
        'fixed acidity', 
        'volatile acidity', 
        'citric acid',
        'residual sugar',
        'chlorides',
        'free sulfur dioxide',
        'total sulfur dioxide',
        'density',
        'pH',
        'sulphates',
        'alcohol' 
    ]
    # Fit the model
    clf.fit(df_train[cols], df_train.quality)
    # Predictions
    train_predictions = clf.predict(df_train[cols])
    test_predictions = clf.predict(df_test[cols])
    # Accuracies 
    train_accuracy = metrics.accuracy_score(
        df_train.quality, train_predictions
    )
    test_accuracy = metrics.accuracy_score(
        df_test.quality, test_predictions
    )
    # Append accuracies 
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Plotting
plt.figure(figsize=(10,5))
sns.set_style("whitegrid")
plt.plot(train_accuracies, label="train accuracy")
plt.plot(test_accuracies, label="test accuracy")
plt.legend(loc="upper left", prop={'size': 15})
plt.xticks(range(0,26,5))
plt.xlabel("max_depth", size=20)
plt.ylabel("accuracy", size=20)
plt.show()

### K-fold using sklearn
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("train.csv")
    # Create a new column called Kfold and fill it with -1
    df["kfold"]=-1
    # Randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    # K-fold class
    kf = model_selection.KFold(n_splits=5)
    # Filling new k-fold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold
    # Save the new csv with kfold column
    df.to_csv("train_folds.csv", index=False)

### Stratified Kfold using sklearn
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("train.csv")
    # Create a new column called Kfold and fill it with -1
    df["kfold"]=-1
    # Randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    # Fetch targets 
    y = df.targets.values
    # K-fold class
    kf = model_selection.StratifiedKFold(n_splits=5)
    # Filling new k-fold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_, 'kfold'] = fold
    # Save the new csv with kfold column
    df.to_csv("train_folds.csv", index=False)

### Checking the distribution of labels for wine dataset
b = sns.countplot(x='quality', data=df)
b.set_xlabel("quality", fontsize=20)
b.set_ylabel("count", fontsize=20)

### Stratified Kfold using sklearn for Regression
import numpy as np
import pandas as pd

from sklearn import datasets 
from sklearn import model_selection

def create_folds(data):
    # Create a new column kfold and fill it with -1
    data["kfold"] = -1
    # Randomize the rows of data
    data = data.sample(frac=1).reset_index(drop=True)
    # Calculate the number of bins by Sturge's rule and take the floor of the value
    num_bins = np.floor(1+np.log2(len(data)))
    # bin targets 
    data.loc[:, "bins"] = pd.cut(data["target"], bins=num_bins, labels=Fasle)
    # kfold class
    kf = model_selection.StratifiedKFold(n_splits=5)
    # Filling kfold column
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f
    # Drop the bins column
    data = data.drop("bins", axis=1)
    # Return dataframe with kfolds
    return data

if __name__ == "__main__":
    # Creating a sample dataset with 15000 samples and 100 features and 1 target
    X, y = datasets.make_regression(
        n_samples=15000, n_features=100, n_targets=1
    )
    # Create dataframe out of our numpy arrays
    df = pd.DataFrame(
        X,
        columns=[f"f_{i}" for i in range(X.shape[1])]
    )
    df.loc[:, "target"] = y
    # create folds
    df = create_folds(df)




#################################################################################################
# Chapter 4: Evaluation Metrics
#################################################################################################

### Custom code of accuracy
def accuracy(y_true, y_pred):
    """
    Function to calculate accuracy 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: accuracy score
    """
    # Counter initialization
    correct_counter = 0
    # loop over all element of y_true and y_pred together using zip
    for yt, yp in zip(y_truem y_pred):
        if yt == yp:
            correct_counter += 1
    # return accuracy
    return correct_counter/len(y_true)

### Accuracy using sklearn
from sklearn import metrics 
l1 = [0,1,1,1,0,0,0,1]
l2 = [0,1,0,1,0,1,0,0]
metrics.accuracy_score(l1, l2)

### TP, FP, FN, TN custom implementation (Only for binary classification)
def true_positive(y_true, y_pred):
    """
    Function to calculate True Positives 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: number of True Positives
    """
    # Initialize 
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
    return tp

def false_positive(y_true, y_pred):
    """
    Function to calculate False Positives 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: number of False Positives
    """
    # Initialize
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
    return fp

def false_negative(y_true, y_false):
    """
    Function to calculate False Negatives  
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: number of False Negatives
    """
    # Initialize 
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1 
    return fn

def true_negative(y_true, y_pred):
    """
    Function to calculate True Negatives 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: number of True Negatives
    """
    # Initialize
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
    return tn

### Accuracy function by using TP,TN,FP,FN
def accuracy_v2(y_true, y_pred):
    """
    Function to calculate accuracy using TP,TN,FP,FN 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: accuracy score
    """
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    accuracy_score = (tp+tn)/(tp+fp+fn+tn)
    return accuracy_score

### Custom Precision function
def precision(y_true, y_pred):
    """
    Function to calculate Precision 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: precision score
    """
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true,y_pred)
    precision = tp/(tp+fp)
    return precision

### Custom Recall function
def recall(y_true, y_pred):
    """
    Function to calculate recall 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: recall score
    """
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    recall = tp/(tp+fn)
    return recall

### Precision-Recall curve
precisions = []
recalls = []
# Some assumed thresholds 
thresholds = [0.04909370, 0.05934905, 0.079377,
              0.08584789, 0.11114267, 0.11639273, 
              0.15952202, 0.17554844, 0.18521942, 
              0.27259048, 0.31620708, 0.33056815, 
              0.39095342, 0.61977213]
# For every threshold value calculate precision and recall
for i in thresholds:
    temp_prediction = [1 if x>=i else 0 for x in y_pred]
    p = precision(y_true, temp_prediction)
    r = recall(y_true, temp_prediction)
    precisions.append(p)
    recalls.append(r)
# Plotting 
plt.figure(figsize=(7,7))
plt.plot(recalls, precisions)
plt.xlabel('Recall', fontsize=15)
plt.ylabel('Precision', fontsize=15)

### Custom F1 score function
def f1(y_true, y_pred):
    """
    Function to calculate F1 score 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: F1 score
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    score = 2*p*r/(p+r)
    return score

### F1 score from sklearn
from sklearn import metrics 
metrics.f1_score(y_true, y_pred)

### Custom TPR function
def tpr(y_true, y_pred):
    """
    Function to calculate True Positive Rate 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: tpr/recall
    """
    return recall(y_true, y_pred)

### Custom FPR function
def fpr(y_true, y_pred):
    """
    Function to calculate False Positive Rate 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: fpr
    """
    fp = false_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    return fp/(fp+tn)

### Calculating and plotting TPR vs FPR
# empty lists to store tpr and fpr values
tpr_list = []
fpr_list = []
# Actual targets 
y_true = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
# Predicted probabilities of a sample being 1
y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 0.3, 0.66, 0.3, 0.2, 0.85, 0.15, 0.99]
# Custom thresholds 
thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0]
# Looping over all thresholds 
for thresh in thresholds:
    # calculating prediction for given threshold 
    temp_pred = [1 if x>=thresh else 0 for x in y_pred]
    # calculate tpr
    temp_tpr = tpr(y_true, temp_pred)
    # calculate fpr
    temp_fpr = fpr(y_true, temp_pred)
    # Appending 
    tpr_list.append(temp_tpr)
    fpr_list.append(temp_fpr)

    










































































 
















    



































