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

# Plotting 
plt.figure(figsize=(7, 7))
plt.fill_between(fpr_list, tpr_list, alpha=0.4)
plt.plot(fpr_list, tpr_list, lw=3)
plt.xlim(0, 1.0)
plt.ylim(0, 1.0)
plt.xlabel('FPR', fontsize=15)
plt.ylabel('TPR', fontsize=15)
plt.show()

### Using sklearn to calculate roc-auc score
from sklearn import metrics 
y_true = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 0.3, 0.66, 0.3, 0.2, 0.85, 0.15, 0.99]

metrics.roc_auc_score(y_true, y_pred)

### How threshold impacts TPR and FPR
tp_list = []
fp_list = []

y_true = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 0.3, 0.66, 0.3, 0.2, 0.85, 0.15, 0.99]

thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0]

for thresh in thresholds:
    temp_pred = [1 if x>= thresh else 0 for x in y_pred]
    temp_tp = true_positive(y_true, temp_pred)
    temp_fp = false_positive(y_true, temp_pred)
    tp_list.append(temp_tp)
    fp_list.append(temp_fp)

### Custom log-loss implementation

import numpy as np

def log_loss(y_true, y_proba):
    """
    Function to calculate Log loss 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: log loss
    """
    # define an epsilon value, this can also be an input, this value is used to clip probabilities 
    epsilon = 1e-15
    # Initialize empty list to store the individual losses
    loss = []
    # Looping 
    for yt, yp in zip(y_true, y_proba):
        # Adjust probabilities, 0 gets converted to 1e-15 and 1 gets converted to 1-1e-15, Why we need it?
        yp = np.clip(yp, epsilon, 1-epsilon)
        # Loss
        temp_loss = -1.0*(yt*np.log(yp) + (1-yt)*np.log(1-yp))
        # Append
        loss.append(temp_loss)
    # Return mean loss
    return np.mean(loss)

### log loss from sklearn
from sklearn import metrics
metrics.log_loss(y_true, y_proba)

### Custom Macro-averaged precision
import numpy as np

def macro_precision(y_true, y_pred):
    """
    Function to calculate Macro-averaged precision 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: macro-averaged precision
    """
    # Find number of classes by taking length of unique values in true list
    num_classes = len(np.unique(y_true))
    # Initialize precision to 0
    precision = 0
    # loop over all classes
    for class_ in range(num_classes):
        # all classes accept current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        # calculate true positive for current class
        tp = true_positive(temp_true, temp_pred)
        # calculate false positive for current class
        fp = false_positive(temp_true, temp_pred)
        # calculate precision for current class
        temp_precision = tp/(tp+fp)
        # keep adding precision for all classes
        precision += temp_precision
    # return avg precision over all classes
    precision /= num_classes
    return precision

### Custom Micro-averaged precision
import numpy as np

def micro_precision(y_true, y_pred):
    """
    Function to calculate Micro-averaged precision 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: micro-averaged precision
    """
    # Find number of classes by taking length of unique values in true list
    num_classes = len(np.unique(y_true))
    # Initialize tp and fp to 0
    tp = 0
    fp = 0
    # loop over all classes
    for class_ in range(num_classes):
        # all classes accept current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        # calculate true positive for current class and update overall tp
        tp += true_positive(temp_true, temp_pred)
        # calculate false positive for current class and update overall fp
        fp += false_positive(temp_true, temp_pred)
    # calculate and return overall precision
    precision = tp/(tp+fp)
    return precision

### Custom Weighted-averaged precision
from collections import Counter
import numpy as np

def weighted_precision(y_true, y_pred):
    """
    Function to calculate Weighted-averaged precision 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: weighted-averaged precision
    """
    # Find number of classes by taking length of unique values in true list
    num_classes = len(np.unique(y_true))
    # create class:sample count dictionary, it looks something like this: {0:20, 1:15, 2:21}
    class_counts = Counter(y_true)
    # Initialize precision to 0
    precision = 0
    # loop over all classes
    for class_ in range(num_classes):
        # all classes accept current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        # calculate tp and fp for current class
        tp = true_positive(temp_true, temp_pred)
        fp = false_positive(temp_true, temp_pred)
        # calculate precision of current class
        temp_precision = tp/(tp+fp)
        # multiply precision with count of samples in class
        weighted_precision = class_counts[class_]*temp_precision
        # add to overall precision
        precision += weighted_precision
    # calculate and return overall precision
    overall_precision = precision/len(y_true)
    return overall_precision

### Macro, micro and weighted precision from sklearn
from sklearn import metrics

y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]
y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]

metrics.precision_score(y_true, y_pred, average="macro")
metrics.precision_score(y_true, y_pred, average="micro")
metrics.precision_score(y_true, y_pred, average="weighted")

### Custom weighted-averaged f1
from collections import Counter
import numpy as np

def weighted_f1(y_true, y_pred):
    """
    Function to calculate Weighted-averaged F1
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: weighted-averaged F1
    """
    # Find number of classes by taking length of unique values in true list
    num_classes = len(np.unique(y_true))
    # create class:sample count dictionary, it looks something like this: {0:20, 1:15, 2:21}
    class_counts = Counter(y_true)
    # Initialize F1 to 0
    f1 = 0
    # loop over all classes
    for class_ in range(num_classes):
        # all classes accept current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        # calculate precision and recall for current class
        p = precision(temp_true, temp_pred)
        r = recall(temp_true, temp_pred)
        # calculate f1 of current class
        if p+r != 0:
            temp_f1 = 2*p*r/(p+r)
        else: 
            temp_f1 = 0
        # multiply precision with count of samples in class
        weighted_f1 = class_counts[class_]*temp_f1
        # add to overall precision
        f1 += weighted_f1
    # calculate and return overall precision
    overall_f1 = f1/len(y_true)
    return overall_f1

### Weighted F1 with sklearn
from sklearn import metrics

y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]
y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]

metrics.f1_score(y_true, y_pred, average="weighted")

### Confusion matrix using sklearn
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import metrics

y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]
y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]
# Get confusion matrix from sklearn
cm = metrics.confusion_matrix(y_true, y_pred)
# Plotting 
plt.figure(figsize=(10,10))
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.set(font_scale=2.5)
sns.heatmap(cm, annot=True, cmap=cmap, cbar=False)
plt.ylabel('Actual Labels', fontsize=20)
plt.xlabel('Predicted Labels', fontsize=20)

### Custom precision at k OR P@k for multi-label classification
def pk(y_true, y_pred, k):
    """
    Function to calculate Precision at k for a single sample
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: precision at a given value k
    """
    # If k is 0, return 0. We should never have this as k is always >= 0
    if k == 0:
        return 0
    # We are interested only in top-k predictions
    y_pred = y_pred[:k]
    # Convert predictions to set
    pred_set = set(y_pred)
    # Convert actual values to set
    true_set = set(y_true)
    # Find common values
    common_values = pred_set.intersection(true_set)
    # return length of common values over k
    return len(common_values)/len(y_pred[:k])

### Custom average precision at k OR AP@k for multi-label classification
def apk(y_true, y_pred, k):
    """
    Function to calculate Average precision at k
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: average precision at a given value k
    """
    # Initialize p@k list of values 
    pk_values = []
    # loop over all k. From 1 to k+1
    for i in range(1, k+1):
        # Calculate P@i and append to list
        pk_values.append(pk(y_true, y_pred, i))
    # If we have no values in the list, return 0
    if len(pk_values) == 0:
        return 0
    # Else, we return the sum of list over length of list
    return sum(pk_values)/len(pk_values)

### Trying AP@k defined above 
y_true = [
    [1, 2, 3],
    [0, 2],
    [1],
    [2, 3],
    [1, 0],
    []
]

y_pred = [
    [0, 1, 2],
    [1],
    [0, 2, 3],
    [2, 3, 4, 0],
    [0, 1, 2],
    [0]
]

for i in range(len(y_true)):
    for j in range(1,4):
        print(
            f"""
            y_true = {y_true[i]},
            y_pred = {y_pred[i]},
            AP@{j} = {apk(y_true[i], y_pred[i], k=j)}
            """
        )

### Custom Mean average precision at k OR MAP@k for multi-label classification
def mapk(y_true, y_pred, k):
    """
    Function to calculate Mean average precision at k
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: mean average precision at a given value k
    """
    # Initialize empty list for apk values
    apk_values = []
    # Loop over all samples
    for i in range(len(y_true)):
        # Store apk values for every sample
        apk_values.append(apk(y_true[i], y_pred[i], k=k))
    # return mean of apk values list
    return sum(apk_values)/len(apk_values)

### Trying MAP@k defined above 
y_true = [
    [1, 2, 3],
    [0, 2],
    [1],
    [2, 3],
    [1, 0],
    []
]

y_pred = [
    [0, 1, 2],
    [1],
    [0, 2, 3],
    [2, 3, 4, 0],
    [0, 1, 2],
    [0]
]

mapk(y_true, y_pred, k=1)
mapk(y_true, y_pred, k=2)
mapk(y_true, y_pred, k=3)
mapk(y_true, y_pred, k=4)

### Custom Mean Absolute Error OR MAE function for regression
import numpy as np

def mean_absolute_error(y_true, y_pred):
    """
    Function to calculate mean absolute error
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: mean absolute error
    """
    # Initialize error at 0
    error = 0
    # Loop over all samples in the true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # Calculate abs error and add to error
        error += np.abs(yt-yp)
    # Return mean error
    return error/len(y_true)

### Custom Mean Squared Error OR MSE function for regression
def mean_squared_error(y_true, y_pred):
    """
    Function to calculate mean squared error
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: mean squared error
    """
    # Initialize error at 0
    error = 0
    # Loop over all samples in the true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # Calculate squared error and add to error
        error += (yt-yp)**2
    # Return mean error
    return error/len(y_true)

### Custom Mean Squared Log Error OR MSLE for regression
def mean_squared_log_error(y_true, y_pred):
    """
    Function to calculate mean squared log error
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: mean squared log error
    """
    # Initialize error at 0
    error = 0
    # Loop over all samples in the true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # Calculate squared log error and add to error
        error += (np.log(1+yt) - np.log(1+yp))**2
    # Return mean error
    return error/len(y_true)

### Custom Mean Percentage Error OR MPE for regression
def mean_percentage_error(y_true, y_pred):
    """
    Function to calculate mean percentage error
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: mean percentage error
    """
    # Initialize error at 0
    error = 0
    # Loop over all samples in the true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # Calculate percentage error and add to error
        error += (yt-yp)/yt
    # Return mean error
    return error/len(y_true)

### Custom Mean Absolute Percentage Error OR MAPE for regression
def mean_absolute_percentage_error(y_true, y_pred):
    """
    Function to calculate mean absolute percentage error
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: mean absolute percentage error
    """
    # Initialize error at 0
    error = 0
    # Loop over all samples in the true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # Calculate abs error and add to error
        error += np.abs(yt-yp)/yt
    # Return mean error
    return error/len(y_true)

### Custom R-squared OR R2 for regression
def r2(y_true, y_pred):
    """
    Function to calculate R-squared score
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: r2 score
    """
    # Calculate the mean value of true values
    mean_true_value = np.mean(y_true)
    # Initialize numerator with 0
    numerator = 0
    # Initialize denominator with 0
    denominator = 0
    # Loop over all true and predicted value
    for yt, yp in zip(y_true, y_pred):
        # update numerator 
        numerator += (yt-yp)**2
        # update denominator
        denominator += (yt-mean_true_value)**2
    # calculate the ratio
    ratio = numerator/denominator
    # return 1-ratio
    return 1-ratio

### Quadratic weighted kappa or Cohen's kappa from sklearn
from sklearn import metrics

y_true = [1, 2, 3, 1, 2, 3, 1, 2, 3]
y_pred = [2, 1, 3, 1, 2, 3, 3, 1, 2]

metrics.cohen_kappa_score(y_true, y_pred, weights="quadratic")
metrics.accuracy_score(y_true, y_pred)

### Custom function for Matthew's Correlation Coefficient for classification problem
def mcc(y_true, y_pred):
    """
    Function to calculate Matthew's Correlation Coefficient
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: MCC score
    """
    tp = true_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)

    numerator = (tp*tn)-(fp*fn)
    denominator = (
        (tp+fp)
        * (fn+tn)
        * (fp+tn)
        * (tp+fn)
    )

    denominator = denominator**0.5
    return numerator/denominator



#################################################################################################
# Chapter 5: Arranging Machine Learning Projects
#################################################################################################

### src/train.py
import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

def run(fold):
    # read the training data with folds
    df = pd.read_csv("../input/mnist_train_folds.csv")
    # training data is where kfold is not equal to provided fold, also note that we reset the index
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    # drop the label column from dataframe and convert it to a numpy array by using .values.
    # target is label column in the dataframe
    x_train = df_train.drop("label", axis=1).values
    y_train = df_train.label.values
    # similarly, for validation, we have
    x_valid = df_valid.drop("label", axis=1).values
    y_valid = df_valid.label.values
    # initialize simple decision tree classifier from sklearn
    clf = tree.DecisionTreeClassifier()
    # fit the model on training data
    clf.fit(x_train, y_train)
    # create predictions for validation samples
    preds = clf.predict(x_valid)
    # calculate & print accuracy 
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")
    # save the model 
    joblib.dump(clf, f"../model/dt_{fold}.bin")

if __name__ == "__main__":
    run(fold=0)
    run(fold=1)
    run(fold=2)
    run(fold=3)
    run(fold=4)

### src/config.py

TRAINING_FILE = "../input/mnist_train_folds.csv"
MODEL_OUTPUT = "../models/"

### src/train.py v2
import os
import config
import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

def run(fold):
    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)
    # training data is where kfold is not equal to provided fold, also note that we reset the index
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    # drop the label column from dataframe and convert it to a numpy array by using .values.
    # target is label column in the dataframe
    x_train = df_train.drop("label", axis=1).values
    y_train = df_train.label.values
    # similarly, for validation, we have
    x_valid = df_valid.drop("label", axis=1).values
    y_valid = df_valid.label.values
    # initialize simple decision tree classifier from sklearn
    clf = tree.DecisionTreeClassifier()
    # fit the model on training data
    clf.fit(x_train, y_train)
    # create predictions for validation samples
    preds = clf.predict(x_valid)
    # calculate & print accuracy 
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")
    # save the model 
    joblib.dump(
        clf,
        os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
    )

if __name__ == "__main__":
    run(fold=0)
    run(fold=1)
    run(fold=2)
    run(fold=3)
    run(fold=4)

### src/train.py v3
import argparse
.
.
.

if __name__ == "__main__":
    # Initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()
    # add the different arguments you need and their type 
    # currently, we only need fold
    parser.add_argument(
        "--fold",
        type=int
    )
    # read the arguments from the command line 
    args = parser.parse_args()
    # run the folds specified by command line arguments 
    run(fold=args.fold)

### shell script
#!/bin/sh
python train.py --fold 0
python train.py --fold 1
python train.py --fold 2
python train.py --fold 3
python train.py --fold 4

### Running shell script
sh run.sh

### src/model_dispatcher.py
from sklearn import tree

models = {
    "decision_tree_gini": tree.DecisionTreeClassifier(
        criterion="gini"
    ),
    "decision_tree_entropy": tree.DecisionTreeClassifier(
        criterion="entropy"
    )
}

### src/train.py v4
import argparse
import os
import config
import model_dispatcher
import joblib
import pandas as pd
from sklearn import metrics

def run(fold, model):
    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)
    # training data is where kfold is not equal to provided fold, also note that we reset the index
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    # drop the label column from dataframe and convert it to a numpy array by using .values.
    # target is label column in the dataframe
    x_train = df_train.drop("label", axis=1).values
    y_train = df_train.label.values
    # similarly, for validation, we have
    x_valid = df_valid.drop("label", axis=1).values
    y_valid = df_valid.label.values
    # fetch the model from model_dispatcher
    clf = model_dispatcher.models[model]
    # fit the model on training data
    clf.fit(x_train, y_train)
    # create predictions for validation samples
    preds = clf.predict(x_valid)
    # calculate & print accuracy 
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")
    # save the model 
    joblib.dump(
        clf,
        os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
    )

if __name__ == "__main__":
    # Initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()
    # add the different arguments you need and their type 
    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )
    # read the arguments from the command line 
    args = parser.parse_args()
    # run the folds specified by command line arguments 
    run(
        fold=args.fold,
        model=args.model
    )

### sample run
python train.py --fold 0 --model decision_tree_gini
python train.py --fold 0 --model decision_tree_entropy

### src/model_dispatcher.py v2
from sklearn import tree
from sklearn import ensemble

models = {
    "decision_tree_gini": tree.DecisionTreeClassifier(
        criterion="gini"
    ),
    "decision_tree_entropy": tree.DecisionTreeClassifier(
        criterion="entropy"
    ),
    "rf": ensemble.RandomForestClassifier()
}

### sample run
python train.py --fold 0 --model rf

###!/bin/sh
python train.py --fold 0 --model rf
python train.py --fold 1 --model rf
python train.py --fold 2 --model rf
python train.py --fold 3 --model rf
python train.py --fold 4 --model rf



#################################################################################################
# Chapter 6: Approaching Categorical Variables
#################################################################################################

### Custom Label Encoding
mapping = {
    "Freeaing": 0,
    "Warm": 1,
    "Cold": 2,
    "Boiling Hot": 3,
    "Hot": 4,
    "Lava Hot": 5
}

import pandas as pd

df = pd.read_csv("../input/cat_train.csv")
df.loc[:, "ord_2"] = df.ord_2.map(mapping)
df.ord_2.value_counts()

### Label Encoding using sklearn
import pandas as pd 
from sklearn import preprocessing
# read the data
df = pd.read_csv("../input/cat_train.csv")
# fill NaN values in ord_2 column
df.loc[:, "ord_2"] = df.ord_2.fillna("NONE")
# initialize LabelEncoder
lbl_enc = preprocessing.LabelEncoder()
# fit label encoder and transform values on ord_2 column
# P.S: do not use this directly. fit first, then transform
df.loc[:, "ord_2"] = lbl_enc.fit_transform(df.ord_2.values)

### Binarize the data, checking size of a matrix 
import numpy as np
# create our example feature matrix
example = np.array(
    [
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 1]
    ]
)
# print size in bytes 
print(example.nbytes)

### Covnerting to Sparse Matrix
import numpy as np
from scipy import sparse

# create our example feature matrix
example = np.array(
    [
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 1]
    ]
)
# convert numpy array to spare CSR matrix
sparse_example = sparse.csr_matrix(example)
# print size of this sparse matrix
print(sparse_example.data.nbytes)

### Total size of a sparse matrix 
print(
    sparse_example.data.nbytes 
    + sparse_example.indptr.nbytes
    + sparse_example.indices.nbytes
)

### Effect on size for a large dataset, benefits of using sparse matrix
import numpy as np 
from scipy import sparse

# number of rows
n_rows = 10000
# number of columns
n_cols = 10000
# create random binray matrix with only 5% value as 1s 
example = np.random.binomial(1, p=0.05, size=(n_rows, n_cols))
# print size in bytes
print(f"Size of dense array: {example.nbytes}")
# convert numpy array to sparse CSR matrix
sparse_example = sparse.csr_matrix(example)
# print size of this sparse matrix
print(f"Size of sparse array: {sparse_example.data.nbytes}")

full_size = (
    sparse_example.data.nbytes 
    + sparse_example.indptr.nbytes
    + sparse_example.indices.nbytes
)
# print full size of this sparse matrix
print(f"Full size of sparse array: {full_size}")

### Converting to one-hot-encoding and check the size
import numpy as np
from scipy import sparse 
# create binary matrix 
example = np.array(
    [
        [0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0]
    ]
)
# print size in bytes
print(f"Size of dense array: {example.nbytes}")
# convert numpy array to sparse CSR matrix 
sparse_example = sparse.csr_matrix(example)
# print size of this sparse matrix
print(f"Size of sparse array: {sparse_example.data.nbytes}")

full_size = (
    sparse_example.data.nbytes 
    + sparse_example.indptr.nbytes
    + sparse_example.indices.nbytes
)
# print full size of this sparse matrix
print(f"Full size of sparse array: {full_size}")

### Large data with one-hot-encoding with sklearn
import numpy as np
from sklearn import preprocessing

# create random 1-d array with 1001 different categories (int)
example = np.random.randint(1000, size=1000000)
# initialize OneHotEncoder from scikit-learn 
# keep sparse = False to get dense array
ohe = preprocessing.OneHotEncoder(sparse=False)
# fit and transform data with dense one-hot-encoder
ohe_example = ohe.fit_transform(example.reshape(-1,1))
# print size in bytes for dense array
print(f"Size of dense array: {ohe_example.nbytes}")
# initialize OneHotEncoder from scikit-learn
# keep sparse = True to get sparse array
ohe = preprocessing.OneHotEncoder(sparse=True)
# fit and transform data wwith sparse one-hot-encoder
ohe_example = ohe.fit_transform(example.reshape(-1,1))
# print size of this sparse matrix
print(f"Size of sparse array: {ohe_example.data.nbytes}")

full_size = (
    ohe_example.data.nbytes 
    + ohe_example.indptr.nbytes
    + ohe_example.indices.nbytes
)
# print full size of this sparse matrix
print(f"Ful size of sparse array: {full_size}")

### Some more ways of dealing with categorical variables and generating new features
df[df.ord_2 == "Boiling Hot"].shape

df.groupby(["ord_2"])["id"].count()

df.groupby(["ord_2"])["id"].transform("count")

df.groupby(
    [
        "ord_1",
        "ord_2"
    ]
)["id"].count().reset_index(name="count")

df["new_feature"] = (
    df.ord_1.astype(str) 
    + "_"
    + df.ord_2.astype(str)
)

df["new_feature"] = (
    df.ord_1.astype(str) 
    + "_"
    + df.ord_2.astype(str)
    + "_"
    + df.ord_3.astype(str)
)

df.ord_2.value_counts()

df.ord_2.fillna("NONE").value_counts()

df.ord_2.value_counts()

df.ord_2.fillna("NONE").value_counts()

### Sample code for applying same transformation to training and test data 
import pandas as pd 
from sklearn import preprocessing
# read training data
train = pd.read_csv("../input/cat_train.csv")
# read test data 
test = pd.read_csv("../input/cat_test.csv")
# create a fake target column for test data since this column doesn't exist 
test.loc[:"target"] = -1
# concatenate both training and test data 
data = pd.concat([train, test]).reset_index(drop=True)
# make a list of features we are interested in, id and target is something we should not encode
features = [x for x in train.columns if x not in ["id", "target"]]
# loop over the features list
for feat in features:
    # create a new instance of LabelEncoder for each feature
    lbl_enc = preprocessing.LabelEncoder()
    # note the trick here
    # since its categorical data, we fillna with a string 
    # and we convert all the data to string type
    # so, no matter its int or float, its converted to string 
    # int/float but categorical!!!
    temp_col = data[feat].fillna("NONE").astype(str).values
    # we can use fit_transform here as we do not 
    # have any extra test data that we need to 
    # transform on separately
    data.loc[:,feat] = lbl_enc.fit_transform(temp_col)
# split the training and test data again
train = data[data.target != -1].reset_index(drop=True)
test = data[data.target == -1].reset_index(drop=True)

### Handling ord_4 column
df.ord_4.fillna("NONE").value_counts()

df.ord_4 = df.ord_4.fillna("NONE")
df.loc[
    df["ord_4"].value_counts()[df["ord_4"]].values < 2000,
    "ord_4"
] = "RARE"
df.ord_4.value_counts()

### Building Models for categorical problem
### src/create_folds.py
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    # read training data
    df = pd.read_csv("../input/cat_train.csv")
    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1
    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    # fetch labels 
    y = df.target.values
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)
    # fill the new kfold column
    for f, (t_,v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    # save the new csv with kfold column
    df.to_csv("../input/cat_train_folds.csv", index=False)

### checking new folds 
import pandas as pd

df = pd.read_csv("../input/cat_train_folds.csv")
df.kfold.value_counts()

### checking target distribution per fold
df[df.kfold==0].target.value_counts()
df[df.kfold==1].target.value_counts()
df[df.kfold==2].target.value_counts()
df[df.kfold==3].target.value_counts()
df[df.kfold==4].target.value_counts()

### ohe_logreg.py
import pandas as pd 
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    # load the full training data with folds
    df = pd.read_csv("../input/cat_train_folds.csv")
    # all columns are features except id, target and kfold columns
    features = [f for f in df.columns if f not in ("id", "target", "kfold")]
    # fill all NaN values with NONE
    # note that I am converting all comumns to "strings"
    # it doesn't matter because all are categories
    for col in features: 
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    # initialize OneHotEncoder from sklearn
    ohe = preprocessing.OneHotEncoder()
    # fit ohe on training + validation features
    full_data = pd.concat(
        [df_train[features], df_valid[features]],
        axis=0
    )
    ohe.fit(full_data[features])
    # transform training data
    x_train = ohe.transform(df_train[features])
    # transform validation data
    x_valid = ohe.transform(df_valid[features])
    # initialize Logistic Regression model
    model = linear_model.LogisticRegression()
    # fit model on training data (ohe)
    model.fit(x_train, df_train.target.values)
    # predict on validation data 
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:,1]
    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    # print auc 
    print(auc)

if __name__ == "__main__":
    # run function for fold = 0
    # we can just replace this number and 
    # run this for any fold
    run(0)

$ python ohe_logreg.py

### ohe_logreg.py v2
.
.
.
    # initialize Logistic Regression model
    model = linear_model.LogisticRegression()
    # fit model on training data (ohe)
    model.fit(x_train, df_train.target.values)
    # predict on validation data 
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:,1]
    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    # print auc
    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)

$ python -W ignore ohe_logreg.py

### lbl_rf.py (Label Encoding and Random Forest)

import pandas as pd
from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    # load the full training data with folds
    df = pd.read_csv("../input/cat_train_folds.csv")
    # all columns are features except id, target and kfold columns
    features = [f for f in df.columns if f not in ("id", "target", "kfold")]
    # fill all NaN values with NONE
    # note that I am converting all comumns to "strings"
    # it doesn't matter because all are categories
    for col in features: 
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
    # now its time to label encode the features 
    for col in features:
        # initialize LabelEncoder for each feature column
        lbl = preprocessing.LabelEncoder()
        # fit label encoder on all data
        lbl.fit(df[col])
        # transform all the data 
        df.loc[:,col] = lbl.transform(df[col])
    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    # get training data
    x_train = df_train[features].values
    # get validation data
    x_valid = df_valid[features].values
    # initialize random forest model
    model = ensemble.RandomForestClassifier(n_jobs=-1)
    # fit model on training data (ohe)
    model.fit(x_train, df_train.target.values)
    # predict on validation data 
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:,1]
    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    # print auc
    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)

$ python lbl_rf.py

### ohe_svd_rf.py (OneHotEncoding with singular value decomposition with random forest)
import pandas as pd
from scipy import sparse
from sklearn import decomposition
from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    # load the full training data with folds
    df = pd.read_csv("../input/cat_train_folds.csv")
    # all columns are features except id, target and kfold columns
    features = [f for f in df.columns if f not in ("id", "target", "kfold")]
    # fill all NaN values with NONE
    # note that I am converting all comumns to "strings"
    # it doesn't matter because all are categories
    for col in features: 
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    # initialize OneHotEncoder from sklearn
    ohe = preprocessing.OneHotEncoder()
    # fit ohe on training + validation features
    full_data = pd.concat(
        [df_train[features], df_valid[features]],
        axis=0
    )
    ohe.fit(full_data[features])
    # transform training data
    x_train = ohe.transform(df_train[features])
    # transform validation data
    x_valid = ohe.transform(df_valid[features])
    # initialize Truncated SVD
    # we are reducing the data to 120 components 
    svd = decomposition.TruncatedSVD(n_components=120)
    # fit SVD on full sparse training data
    full_sparse = sparse.vstack((x_train, x_valid))
    svd.fit(full_sparse)
    # transform sparse training data
    x_train = svd.transform(x_train)
    # transform sparse validation data
    x_valid = svd.transform(x_valid)
    # initialize random forest model
    model = ensemble.RandomForestClassifier(n_jobs=-1)
    # fit model on training data (ohe)
    model.fit(x_train, df_train.target.values)
    # predict on validation data 
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:,1]
    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    # print auc
    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)

$ python ohe_svd_rf.py

### lbl_xgb.py
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    # load the full training data with folds
    df = pd.read_csv("../input/cat_train_folds.csv")
    # all columns are features except id, target and kfold columns
    features = [f for f in df.columns if f not in ("id", "target", "kfold")]
    # fill all NaN values with NONE
    # note that I am converting all comumns to "strings"
    # it doesn't matter because all are categories
    for col in features: 
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
    # now its time to label encode the features 
    for col in features:
        # initialize LabelEncoder for each feature column
        lbl = preprocessing.LabelEncoder()
        # fit label encoder on all data
        lbl.fit(df[col])
        # transform all the data 
        df.loc[:,col] = lbl.transform(df[col])
    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    # get training data
    x_train = df_train[features].values
    # get validation data
    x_valid = df_valid[features].values
    # initialize xgboost model
    model = xgb.XGBClassifier(
        n_jobs=-1,
        max_depth=7,
        n_estimators=200
    )
    # fit model on training data (ohe)
    model.fit(x_train, df_train.target.values)
    # predict on validation data 
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:,1]
    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    # print auc
    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)

$ python lbl_xgb.py

### US adult census data
import pandas as pd
df = pd.read_csv("../input/adult.csv")
df.income.value_counts()

### ohe_logres.py
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    # load the full training data with folds
    df = pd.read_csv("../input/adult_folds.csv")
    # list of numerical columns
    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]
    # drop numerical columns
    df = df.drop(num_cols, axis=1)
    # map targets to 0s and 1s
    target_mapping = {
        "<=50K":0,
        ">50K":1
    }
    df.loc[:,"income"] = df.income.map(target_mapping)
    # all columns are features execpt income and kfold columns
    features = [
        f for f in df.columns if f not in ("kfold", "income")
    ]
    # fill all NaN values with NONE
    # note that I am converting all columns to "strings"
    # it doesn't matter because all are categories
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
    # get training data using folds 
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    # initialize OneHotEncoder from sklearn
    ohe = preprocessing.OneHotEncoder()
    # fit ohe on training + validation features
    full_data = pd.concat(
        [df_train[features], df_valid[features]],
        axis=0
    )
    ohe.fit(full_data[features])
    # transform training data
    x_train = ohe.transform(df_train[features])
    # transform validation data
    x_valid = ohe.transform(df_valid[features])
    # initialize the logistic regression model
    model = linear_model.LogisticRegression()
    # fit model on training data (ohe)
    model.fit(x_train, df_train.income.values)
    # predict on validation data 
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:,1]
    # get ROC AUC score
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)
    # print auc
    print(f"Fold={fold}, AUC={auc}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)

$ python -W ignore ohe_logreg.py

### lbl_xgb.py
import pandas as pd 
import xgboost as xgb
from sklearn import metrics 
from sklearn import preprocessing

def run(fold):
    # load the full training data with folds
    df = pd.read_csv("../input/adult_folds.csv")
    # list of numerical columns
    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]
    # drop numerical columns
    df = df.drop(num_cols, axis=1)
    # map targets to 0s and 1s
    target_mapping = {
        "<=50K":0,
        ">50K":1
    }
    df.loc[:,"income"] = df.income.map(target_mapping)
    # all columns are features execpt income and kfold columns
    features = [
        f for f in df.columns if f not in ("kfold", "income")
    ]
    # fill all NaN values with NONE
    # note that I am converting all columns to "strings"
    # it doesn't matter because all are categories
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
    # now its time to label ecode the features
    for col in features:
        # initialize LabelEncoder for each feature column
        lbl = preprocessing.LabelEncoder()
        # fit label encoder on all data
        lbl.fit(df[col])
        # transform all the data 
        df.loc[:,col] = lbl.transform(df[col])
    # get training data using folds 
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    # get training data
    x_train = df_train[features].values
    # get validation data
    x_valid = df_valid[features].values
    # initialize the xgb model
    model = xgb.XGBClassifier(
        n_jobs=-1
    )
    # fit model on training data (ohe)
    model.fit(x_train, df_train.income.values)
    # predict on validation data 
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:,1]
    # get ROC AUC score
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)
    # print auc
    print(f"Fold={fold}, AUC={auc}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)

$ python lbl_xgb.py

### lbl_xgb_num.py
import pandas as pd 
import xgboost as xgb
from sklearn import metrics 
from sklearn import preprocessing

def run(fold):
    # load the full training data with folds
    df = pd.read_csv("../input/adult_folds.csv")
    # list of numerical columns
    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]
    # map targets to 0s and 1s
    target_mapping = {
        "<=50K":0,
        ">50K":1
    }
    df.loc[:,"income"] = df.income.map(target_mapping)
    # all columns are features execpt income and kfold columns
    features = [
        f for f in df.columns if f not in ("kfold", "income")
    ]
    # fill all NaN values with NONE
    # note that I am converting all columns to "strings"
    # it doesn't matter because all are categories
    for col in features:
        # do not encode the numerical columns
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna("NONE")
    # now its time to label ecode the features
    for col in features:
        if col not in num_cols:
            # initialize LabelEncoder for each feature column
            lbl = preprocessing.LabelEncoder()
            # fit label encoder on all data
            lbl.fit(df[col])
            # transform all the data 
            df.loc[:,col] = lbl.transform(df[col])
    # get training data using folds 
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    # get training data
    x_train = df_train[features].values
    # get validation data
    x_valid = df_valid[features].values
    # initialize the xgb model
    model = xgb.XGBClassifier(
        n_jobs=-1
    )
    # fit model on training data (ohe)
    model.fit(x_train, df_train.income.values)
    # predict on validation data 
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:,1]
    # get ROC AUC score
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)
    # print auc
    print(f"Fold={fold}, AUC={auc}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)

$ python lbl_xgb_num.py

### lbl_xgb_num_feat.py
import itertools
import pandas as pd 
import xgboost as xgb
from sklearn import metrics 
from sklearn import preprocessing

def feature_engineering(df, cat_cols):
    """
    This function is used for feature engineering 
    :param df: the pandas dataframe with train/test data
    :param cat_cols: list of categorical columns
    :return: dataframe with new features
    """
    # this will create all 2-combinations of values
    # in this list 
    # for example:
    # list(itertools.combinations([1,2,3],2)) will return
    # [(1,2), (1,3), (2,3)]
    combi = list(itertools.combinations(cat_cols, 2))
    for c1, c2 in combi:
        df.loc[
            :,
            c1 + "_" + c2
        ] = df[c1].astype(str) + "_" + df[c2].astype(str)
    return df 

def run(fold):
    # load the full training data with folds
    df = pd.read_csv("../input/adult_folds.csv")
    # list of numerical columns
    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]
    # map targets to 0s and 1s
    target_mapping = {
        "<=50K":0,
        ">50K":1
    }
    df.loc[:,"income"] = df.income.map(target_mapping)
    # list of categorical columns for feature engineering 
    cat_cols = [
        c for c in df.columns if c not in num_cols
        and c not in ("kfold", "income")
    ]
    # add new features 
    df = feature_engineering(df, cat_cols)
    # all columns are features execpt income and kfold columns
    features = [
        f for f in df.columns if f not in ("kfold", "income")
    ]
    # fill all NaN values with NONE
    # note that I am converting all columns to "strings"
    # it doesn't matter because all are categories
    for col in features:
        # do not encode the numerical columns
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna("NONE")
    # now its time to label ecode the features
    for col in features:
        if col not in num_cols:
            # initialize LabelEncoder for each feature column
            lbl = preprocessing.LabelEncoder()
            # fit label encoder on all data
            lbl.fit(df[col])
            # transform all the data 
            df.loc[:,col] = lbl.transform(df[col])
    # get training data using folds 
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    # get training data
    x_train = df_train[features].values
    # get validation data
    x_valid = df_valid[features].values
    # initialize the xgb model
    model = xgb.XGBClassifier(
        n_jobs=-1
    )
    # fit model on training data (ohe)
    model.fit(x_train, df_train.income.values)
    # predict on validation data 
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:,1]
    # get ROC AUC score
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)
    # print auc
    print(f"Fold={fold}, AUC={auc}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)

$ python lbl_xgb_num_feat.py

### target_encoding.py (need to check again!)
import copy
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
import xgboost as xgb

def mean_target_encoding(data):
    # make a copy of dataframe
    df = copy.deepcopy(data)
    # list of numerical cols
    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]
    # map targets to 0s and 1s
    target_mapping = {
        "<=50K":0,
        ">50K":1
    }
    df.loc[:, "income"] = df.income.map(target_mapping)
    # all columns are features except income and kfold columns
    features = [
        f for f in df.columns if f not in ("kfold", "income")
    ]
    # fill all NaN values with NONE
    # note that I am converting all columns to "strings"
    # it doesn't matter because all are categories
    for col in features:
        # do not encode the numerical columns
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna("NONE")
    # now its time to label ecode the features
    for col in features:
        if col not in num_cols:
            # initialize LabelEncoder for each feature column
            lbl = preprocessing.LabelEncoder()
            # fit label encoder on all data
            lbl.fit(df[col])
            # transform all the data 
            df.loc[:,col] = lbl.transform(df[col])
    # a list to store 5 validation dataframes
    encoded_dfs = []
    # go over all folds
    for fold in range(5):
        # fetch training and validation data
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        # for all feature columns, i.e. categorical columns
        for column in features: 
            if col not in num_cols:
                # create dict of category:mean target
                mapping_dict = dict(
                    df_train.groupby(column)["income"].mean()
                )
                # column_enc is the new column we have with mean encoding 
                df_valid.loc[
                    :, column + "_enc"
                ] = df_valid[column].map(mapping_dict)
        # append to our list of encoded validation dataframes
        encoded_dfs.append(df_valid)
    # create full data frame again and return
    encoded_df = pd.concat(encoded_dfs, axis=0)
    return encoded_df

def run(fold):
    # note that folds are same as before
    # get training data using folds 
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    # all columns are features except income and kfold columns
    features = [
        f for f in df.columns if f not in ("kfold", "income")
    ]
    # get training data
    x_train = df_train[features].values
    # get validation data
    x_valid = df_valid[features].values
    # initialize the xgb model
    model = xgb.XGBClassifier(
        n_jobs=-1,
        max_depth = 7
    )
    # fit model on training data (ohe)
    model.fit(x_train, df_train.income.values)
    # predict on validation data 
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:,1]
    # get ROC AUC score
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)
    # print auc
    print(f"Fold={fold}, AUC={auc}")

if __name__ == "__main__":
    # read data 
    df = pd.read_csv("../input/adult_folds.csv")
    # create mean target encoded categories and 
    # munge data
    df = mean_target_encoding(df)
    # run training and validation for 5 folds
    for fold_ in range(5):
        run(df, fold_)

### entity_embeddings.py
import os 
import gc
import joblib
import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing
from tensorflow.keras import layers 
from tensorflow.keras import optimizers 
from tensorflow.keras.models import Model, load_model
from tensorflow.kears import callbacks 
from tensorflow.kears import backend as K
from tensorflow.kears import utils

def create_model(data, catcols):
    """
    This function returns a compiled tf.keras model
    for entity embeddings 
    :param data: this is a pandas dataframe
    :param catcols: list of categorical column names
    :return: compiled tf.keras model
    """
    # init list of inputs for embeddings 
    inputs = []
    # init list of outputs for embeddings 
    outputs = []
    # loop over all categorical columns
    for c in catcols:
        # find the number of unique values in the column
        num_unique_values = int(data[c].nunique())
        # simple dimension of embedding calculator 
        # min size is half of the number of unique values
        # max size is 50. max size depends on the number of unique
        # categories too. 50 is quite sufficient most of the times
        # but if you have millions of unique values, you might need
        # a larger dimension
        embed_dim = int(min(np.ceil((num_unique_values)/2), 50))
        # simple keras input layer with size 1
        inp = layers.Input(shape=(1,))
        # add embedding layer to raw input
        # embedding size is always one more than unique values in input
        out = layers.Embedding(
            num_unique_values+1, embed_dim, name=c
        )(inp)
        # 1-d spatial dropout is the standard for embedding layers 
        # you can use it NLP tasks too
        out = layers.SpatialDropout1D(0.3)(out)
        # reshape the input to the dimension of embedding 
        # this becomes our output layer for current feature 
        out = layers.Reshape(target_shape=(embed_dim,))(out)
        # add input to input list 
        inputs.append(inp)
        # add output to output list 
        outputs.append(out)
    # concatenate all output layers 
    x = layers.Concatenate()(outputs)
    # add a batchnorm layer.
    # from here, everything is up to you
    # you can try different architectures 
    # this is the architecture I like to use 
    # if you have numerical features, you should add 
    # them here or in concatenate layer
    x = layers.BatchNormalization()(x)
    # a bunch of dense layers with dropout.
    # start with 1 or two layers only
    x = layers.Dense(300, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(300, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    # using softmax and treating it as a two class problem
    # you can also use sigmoid, then you need to use only one 
    # output class
    y = layers.Dense(2, activation="softmax")(x)
    # create final model
    model = Model(inputs=inputs, outputs=y)
    # compile the model
    # we use adam and binary cross entropy.
    # feel free to use something else and see how model behaves
    model.compile(loss="binary_crossentropy", optimizer="adam")
    return model

def run(fold):
    # load the full training data with folds 
    df = pd.read_csv("../input/cat_train_folds.csv")
    # all columns are features except id, target and kfold columns
    features = [
        f for f in df.columns if f not in ("id", "target", "kfold")
    ]    
    # fill all NaN values with NONE
    # note that I am converting all columns to "strings"
    # it doesn't matter because all are categories 
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
    # encode all features with label encoder individually 
    # in a live setting you need to save all label encoders
    for feat in features:
        lbl_enc = preprocessing.LabelEncoder()
        df.loc[:,feat] = lbl_enc.fit_transform(df[feat].values)
    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # get validation data using folds 
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    # create tf.keras model
    model = create_model(df, features)
    # our features are lists of lists 
    xtrain = [
        df_train[features].values[:,k] for k in range(len(features))
    ]
    xvalid = [
        df_valid[features].values[:,k] for k in range(len(features))
    ]
    # fetch target columns 
    ytrain = df_train.target.values
    yvalid = df_valid.target.values
    # convert target columns to categories 
    # this is just binarization
    ytrain_cat = utils.to_categorical(ytrain)
    yvalid_cat = utils.to_categorical(yvalid)
    # fit the model
    model.fit(
        xtrain,
        ytrain_cat,
        validation_data = (xvalid, yvalid_cat),
        verbose=1,
        batch_size=1024,
        epochs=3
    )
    # generate validation predictions 
    valid_preds = model.predict(xvalid)[:,1]
    # print roc auc score
    print(metrics.roc_auc_score(yvalid, valid_preds))
    # clear session to free up some GPU memory
    K.clear_session()


if __name__ == "__main__":
    run(0)
    run(1)
    run(2)
    run(3)
    run(4)




#################################################################################################
# Chapter 7: Feature Engineering 
#################################################################################################

### dealing with date time column and creating features out of it.
df.loc[:, "year"] = df["datetime_column"].dt.year
df.loc[:, "weekofyear"] = df["datetime_column"].dt.weekofyear
df.loc[:, "month"] = df["datetime_column"].dt.month
df.loc[:, "dayofweek"] = df["datetime_column"].dt.dayofweek
df.loc[:, "weekend"] = (df.datetime_column.dt.weekday >= 5).astype(int)
df.loc[:, "hour"] = df["datetime_column"].dt.hour

### sample features from date time column
import pandas as pd 
# create a series of datetime with a frequency of 10 hours 
s = pd.date_range('2020-01-06', '2020-01-10', freq='10H').to_series()
# create some features based on datetime
features = {
    "dayofweek": s.dt.dayofweek.values,
    "dayofyear": s.dt.dayofyear.values,
    "hour": s.dt.hour.values,
    "is_leap_year": s.dt.is_leap_year.values,
    "quarter": s.dt.quarter.values,
    "weekofyear": s.dt.weekofyear.values
}

### Using pandas to create aggregate features 
def generate_features(df):
    # create a bunch of features using the date column
    df.loc[:, "year"] = df["date"].dt.year
    df.loc[:, "weekofyear"] = df["date"].dt.weekofyear
    df.loc[:, "month"] = df["date"].dt.month
    df.loc[:, "dayofweek"] = df["date"].dt.dayofweek
    df.loc[:, "weekend"] = (df["date"].dt.weekday >= 5).astype(int)
    # create an aggregate dictionary
    aggs = {}
    # for aggregation by month, we calculate the 
    # number of unique month values and also the mean
    aggs["month"] = ["nunique", "mean"]
    aggs["weekofyear"] = ["nunique", "mean"]
    # we aggregate by num1 and calculate sum, max, min
    # and mean values of this column
    aggs["num1"] = ["sum", "max", "min", "mean"]
    # for customer_id, we calculate the total count
    aggs["customer_id"] = ["size"]
    # again for customer_id, we calculate the total unique
    aggs["customer_id"] = ["nunique"]
    # we group by customer_id and calculate the aggregates
    agg_df = df.groupby("customer_id").agg(aggs)
    agg_df = agg_df.reset_index()
    return agg_df

### creating some features like mean, min, max, std, var, percentile etc.
import numpy as np

feature_dict = {}
# calculate mean 
feature_dict["mean"] = np.mean(x)
# calculate max
feature_dict["max"] = np.max(x)
# calculate min
feature_dict["min"] = np.min(x)
# calculate standard deviation 
feature_dict["std"] = np.std(x)
# calculate variance 
feature_dict["var"] = np.var(x)
# peak-to-peak
feature_dict["ptp"] = np.ptp(x)
# percentile features 
feature_dict["percentile_10"] = np.percentile(x, 10)
feature_dict["percentile_60"] = np.percentile(x, 60)
feature_dict["percentile_90"] = np.percentile(x, 90)
# quantile features 
feature_dict["quantile_5"] = np.percentile(x, 5)
feature_dict["quantile_95"] = np.percentile(x, 95)
feature_dict["quantile_99"] = np.percentile(x, 99)

### features based on tsfresh library 
from tsfresh.feature_extraction import feature_calculators as fc
# tsfresh based features 
feature_dict["abs_energy"] = fc.abs_energy(x)
feature_dict["count_above_mean"] = fc.count_above_mean(x)
feature_dict["count_below_mean"] = fc.count_below_mean(x)
feature_dict["mean_abs_change"] = fc.mean_abs_change(x)
feature_dict["mean_change"] = fc.mean_change(x)

### Simple way of creating polynomial features using sklearn
import numpy as np
# generate a random dataframe with
# 2 columns and 100 rows 
df = pd.DataFrame(
    np.random.rand(100, 2),
    columns = [f"f_{i}" for i in range(1,3)]
)

from sklearn import preprocessing
# initialize polynomial features class object
# for two-degree polynomial features 
pf = preprocessing.PolynomialFeatures(
    degree = 2,
    interaction_only = False,
    include_bias = False 
)
# fit to the features 
pf.fit(df)
# create polynomial features 
poly_feats = pf.transform(df)
# create a dataframe with all the features 
num_feats = poly_feats.shape[1]
df_transformed = pd.DataFrame(
    poly_feats,
    columns=[f"f_{i}" for i in range(1, num_feats+1)]
)

### Creating bins out of numerical columns
# 10 bins
df["f_bin_10"] = pd.cut(df["f_1"], bins=10, labels=False)
# 100 bins 
df["f_bin_100"] = pd.cut(df["f_1"], bins=100, labels=False)

# Applying log transformation to column having high variance 
df.f_3.var()
df.f_3.apply(lambda x: np.log(1+x)).var()

### Using KNN imputer to impute the missing values
import numpy as np
from sklearn import impute
# create a random numpy array with 10 samples
# and 6 features and values ranging from 1 to 15
X = np.random.randint(1, 15, (10, 6))
# convert the array to float 
X = X.astype(float)
# randomly assign 10 elements to NaN (missing)
X.ravel()[np.random.choice(X.size, 10, replace=False)] = np.nan
# use 3 nearest neighbours to fill na values 
knn_imputer = impute.KNNImputer(n_neighbors=2)
knn_imputer.fit_transform(X)




#################################################################################################
# Chapter 8: Feature Selection 
#################################################################################################

### removing low variance columns from the dataset
from sklearn.feature_selection import VarianceThreshold
data = ...
var_thresh = VarianceThreshold(threshold=0.1)
transformed_data = var_thresh.fit_transform(data)
# transformed data will have all columns with variance less than 0.1 removed

### removing features with high correlations 
import pandas as pd
from sklearn.datasets import fetch_california_housing 
# fetch a regression dataset
data = fetch_california_housing()
X = data["data"]
col_names = data["feature_names"]
y = data["target"]
# convert to pandas dataframe
df = pd.DataFrame(X, columns=col_names)
# introduce a highly correlated column
df.loc[:, "MedInc_Sqrt"] = df.MedInc.apply(np.sqrt)
# get correlation matrix (pearson)
df.corr()

### Univariate feature selection
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile

class UnivariateFeatureSelection:
    def __init__(self, n_features, problem_type, scoring):
        """
        Custom univariate feature selection wrapper on
        different univariate feature selection model from
        sklearn.
        :param n_features: SelectPercentile if float else SelectKBest
        :param problem_type: classification or regression
        :param scoring: scoring function, string
        """
        # for a given problem type, threre are only
        # a few valid scoring methods
        # you can extend this with your own custom
        # methods if you wish
        if problem_type == "classification":
            valid_scoring = {
                "f_classif": f_classif,
                "chi2": chi2,
                "mutual_info_classif": mutual_info_classif
            }
        else:
            valid_scoring = {
                "f_regression": f_regression,
                "mutual_info_regression": mutual_info_regression
            }
        # raise exception if we do not have a valid scoring method
        if scoring not in valid_scoring:
            raise Exception("Invalid scoring function")
        # if n_features is int, we use selectkbest
        # if n_features is float, we use selectpercentile
        # please note that it is int in both cases in sklearn
        if isinstance(n_features, int):
            self.selection = SelectKBest(
                valid_scoring[scoring],
                k = n_features
            )
        elif isinstance(n_features, float):
            self.selection = SelectPercentile(
                valid_scoring[scoring],
                percentile = int(n_features*100)
            )
        else:
            raise Exception("Invalid type of feature")
    # same fit function
    def fit(self, X, y):
        return self.selection.fit(X,y)
    # same transform function
    def transform(self, X):
        return self.selection.transform(X)
    # same fit_transform function
    def fit_transform(self, X, y):
        return self.selection.fit_transform(X, y)

### Using above UnivariateFeatureSelection class
ufs = UnivariateFeatureSelection(
    n_features=0.1,
    problem_type="regression",
    scoring="f_regression"
)
ufs.fit(X, y)
X_transformed = ufs.transform(X)

### Greedy Feature Selection
# greedy.py
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.datasets import make_classification

class GreedyFeatureSelection:
    """
    A simple and custom class for greedy feature selection.
    You will need to modigy it quite a bit to make it suitable
    for your dataset.
    """
    def evaluate_score(self, X, y):
        """
        This function evaluates model on data and returns
        Area Under ROC Curve (AUC)
        NOTE: We fit the data and calculate AUC on same data.
        WE ARE OVERFITTING HERE.
        But this is also a way to achieve greedy selection.
        k-fold will take k times longer.

        If you want to implement it in really correct way,
        calculate OOF AUC and return mean AUC over k folds.
        This requires only a few lines of change and has been
        shown a few times in this book.

        :param X: training data
        :param y: targets
        :return : overfitted area under the ROC curve
        """
        # fit the logistic regression model,
        # and calculate AUC on same data
        # again: BEWARE
        # you can choose any model that suits your data
        model = linear_model.LogisticRegression()
        model.fit(X, y)
        predictions = model.predict_proba(X)[:, 1]
        auc = metrics.roc_auc_score(y, predictions)
        return auc

    def _feature_selection(self, X, y):
        """
        This function does the actual greedy selection
        :param X: data, numpy array
        :param y: targets, numpy array
        :return: (best scores, best features)
        """
        # initialize good features list
        # and best scores to keep track of both
        good_features = []
        best_scores = []
        # calculate the number of features
        num_features = X.shape[1]
        # infinite loop
        while True:
            # initialize best feature and score of this loop
            this_feature = None
            best_score = 0
            # loop over all features
            for features in range(num_features):
                # if feature is already in good features,
                # skip this for loop
                if feature in good_features:
                    continue
                # selected features are all good features till now
                # and current feature
                selected_features = good_features + [feature]
                # remove all other features from data
                xtrain = X[:, selected_features]
                # calculate the score, in our case, AUC
                score = self.evaluate_score(xtrain, y)
                # if score is greater than the best score
                # of this loop, change best score and best feature
                if score > best_score:
                    this_feature = feature
                    best_score = score
            # if we have selected a feature, add it
            # to the good feature list and update best scores list
            if this_feature != None
                good_features.append(this_feature)
                best_scores.append(best_score)
            # if we didn't improve during the last two rounds,
            # exit the while loop
            if len(best_scores) > 2:
                if best_scores[-1] < best_scores[-2]:
                    break
        # return best scores and good features
        # why do we remove the last data point?
        return best_scores[:-1], good_features[:-1]

    def __call__(self, X, y):
        """
        Call function will call the class on a set of arguments
        """
        # select featues, return scores and selected indices
        scores, features = self._feature_selection(X, y)
        # transform data with selected features
        return X[:, features], scores

if __name__ == "__main__":
    # generate binary classification data
    X, y = make_classification(n_samples=1000, n_features=100)
    # tranform data by greedy feature selection
    X_transformed, scores = GreedyFeatureSelection()(X, y)

### Recursive feature elimination
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
# fetch a regression dataset
data = fetch_california_housing()
X = data["data"]
col_names = data["feature_names"]
y = data["target"]
# initialize the model
model = LinearRegression()
# initialize RFE
rfe = RFE(
    estimator=model,
    n_features_to_select=3
)
# fit RFE
rfe.fit(X, y)
# get the transformed data with
# selected columns
X_transformed = rfe.transform(X)

### Removing features based on value of feature of importance
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
# fetch a regression dataset
# in diabetes data we predict diabetes progression
# after one year based on some features
data = load_diabetes()
X = data["data"]
col_names = data["feature_names"]
y = data["target"]
# initialize the model
model = RandomForestRegressor()
# fit the model
model.fit(X,y)
# Plotting feature of importance
importance = model.feature_importances_
idxs = np.argsort(importances)
plt.title("Feature Importances")
plt.barh(range(len(idxs)), importances[idxs], align="center")
plt.yticks(range(len(idxs)), [col_names[i] for i in idxs])
plt.xlabel("Random Forest Feature Importance")
plt.show()

### Using sklearn's SelectFromModel to select the features
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
# fetch a regression dataset
# in diabetes data we predict diabetes progression
# after one year based on some features
data = load_diabetes()
X = data["data"]
col_names = data["feature_names"]
y = data["target"]
# initialize the model
model = RandomForestRegressor()
# select from the model
sfm = SelectFromModel(estimator=model)
X_transformed = sfm.fit_transform(X, y)
# see which features were selected
support = sfm.get_support()
# get feature names
print([
    x for x, y in zip(col_names, support) if y == True
])



#################################################################################################
# Chapter 9: Hyperparameter Optimization
#################################################################################################

### Simple understanding of hyperparameter optimization
# define the best accuracy to be 0
# if you choose loss as a metric,
# you can make best loss to be inf (np.inf)
best_accuracy = 0
best_parameters = {"a":0, "b":0, "c":0}
# loop over all values for a, b & c
for a in range(1,11):
    for b in range(1,11):
        for c in range(1,11):
            # initialize model with current parameters
            model = MODEL(a, b, c)
            # fit the model
            model.fit(training_data)
            # make predictions
            preds = model.predict(validation_data)
            # calculate accuracy
            accuracy = metrics.accuracy_score(targets, preds)
            # save params if current accuracy
            # is greater than best accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_parameters["a"] = a
                best_parameters["b"] = b
                best_parameters["c"] = c

### rf_grid_search.py
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

if __name__ == "__main__":
    # raed the training data
    df = pd.read_csv("./input/mobile_train.csv")
    # features are all columns without price_range
    # note that there is no id column in this dataset
    # here we have training features
    X = df.drop("price_range", axis=1).values
    # and the targets
    y = df.price_range.values
    # define the model here
    # i am using random forest with n_jobs = -1
    # n_jobs = -1 => use all cores
    classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    # define a grid of parameters
    # this can be a dictionary or a list of
    # dictionaries
    param_grid = {
        "n_estimators": [100, 200, 250, 300, 400, 500],
        "max_depth": [1, 2, 5, 7, 11, 15],
        "criterion": ["gini", "entropy"]
    }
    # initialize grid search
    # estimator is the model that we have defined
    # param_grid is the grid of parameters
    # we use accuracy as out metric. you can define your own.
    # higher value of verbose implies a lot of details are printed
    # cv=5 means that we are using 5 fold cv (not stratified)
    model = model_selection.GridSearchCV(
        estimator=classifier,
        param_grid = param_grid,
        scoring = "accuracy",
        verbose = 10,
        n_jobs= -1,
        cv = 5
    )
    # fit the model and extract best score
    model.fit(X, y)
    print(f"Best score: {model.best_score_}")
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")

### rf_random_search.py
.
.
.
if __name__ == "__main__":
    .
    .
    .
    # define the model here
    # i am using random forest with n_jobs = -1
    # n_jobs = -1 => use all cores
    classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    # define a grid of parameters
    # this can be a dictionary or a list of
    # dictionaries
    param_grid = {
        "n_estimators": np.arange(100, 1500, 100),
        "max_depth": np.arange(1, 31),
        "criterion": ["gini", "entropy"]
    }
    # initialize random search
    # estimator is the model that we have defined
    # param_distributions is the grid/distribution of parameters
    # we use accuracy as our metric. you can define your own.
    # higher value of verbose implies a lot of details are printed
    # cv=5 means that we are using 5 fold cv (not stratified)
    # n_iter is the number of iterations we want
    # if param_distributions has all the values as list,
    # random search will be done by sampling without replacement
    # if any of the parameters come from a distribution,
    # random search uses sampling with replacement
    model = model_selection.RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_grid,
        n_iter=20,
        scoring="accuracy",
        verbose=10,
        n_jobs=-1,
        cv=5
    )
    # fit the model and extract best score
    model.fit(X, y)
    print(f"Best score: {model.best_score_}")
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")

### pipeline_search.py
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def quadratic_weighted_kappa(y_true, y_pred):
    """
    Craete a wrapper for Cohen's kappa
    with quadratic weights
    """
    return metrics.cohen_kappa_score(
        y_true,
        y_pred,
        weights = "quadratic"
    )

if __name__ == "__main__":
    # load the training file
    train = pd.read_csv("../input/train.csv")
    # we don't need ID columns
    idx = test.id.values.astype(int)
    train = train.drop("id", axis=1)
    test = test.drop("id", axis=1)
    # create labels. drop useless columns
    y = train.relevance.values
    # do some lambda magic on text columns
    traindata = list(
        train.apply(lambda x: "%s %s" % (x["text1"], x["text2"]), axis=1)
    )
    testdata = list(
        test.apply(lambda x: "%s %s" % (x["text1"], x["text2"]), axis=1)
    )
    # tfidf vectorizer
    tfv = TfidfVectorizer(
        min_df = 3,
        max_features = None,
        strip_accents = "unicode",
        analyzer = "word",
        token_pattern = r"\w{1,}",
        ngram_range = (1, 3),
        use_idf = 1,
        smooth_idf = 1,
        sublinear_tf = 1,
        stop_words = "english"
    )
    # Fit TFIDF
    tfv.fit(traindata)
    X = tfv.transform(traindata)
    X_test = tfv.transform(testdata)
    # Initialize SVD
    svd = TruncatedSVD()
    # Initialize the standard scaler
    scl = StandardScaler()
    # We will use SVM here..
    svm_model = SVC()
    # Create the pipeline
    clf = pipeline.Pipeline(
        [
            ("svd", svd),
            ("scl", scl),
            ("svm", svm_model)
        ]
    )
    # Create a parameter grid to search for
    # best parameters for everything in the pipeline
    param_grid = {
        "svd__n_components": [200, 300],
        "svm__C": [10, 12]
    }
    # Kappa Scorer
    kappa_scorer = metrics.make_scorer(
        quadratic_weighted_kappa,
        greater_is_better = True
    )
    # Initialize Grid Search Model
    model = model_selection.GridSearchCV(
        estimator = clf,
        param_grid = param_grid,
        scoring = kappa_scorer,
        verbose = 10,
        n_jobs = -1,
        refit = True,
        cv = 5
    )
    # Fit Grid Search Model
    model.fit(X, y)
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    # Get best model
    best_model = model.best_estimator_
    # Fir model with best parameters optimized for QWK
    best_model.fit(X, y)
    preds = best_model.predict(...)

### rf_gp_minimize.py
import numpy as np
import pandas as pd
from functools import partial
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from skopt import gp_minimize
from skopt import space

def optimize(params, param_names, x, y):
    """
    The main optimization function.
    This function takes all the arguments from the search space
    and training features and targets. It then initializes
    the models by setting the chosen parameters and runs
    cross-validation and returns a negative accuracy score
    :param params: list of params from gp_minimize
    :param param_names: list of param names. order is important!
    :param x: training data
    :param y: labels/targets
    :return: negative accuracy after 5 folds
    """
    # convert params to dictionary
    params = dict(zip(param_names, params))
    # initialize model with current parameters
    model = ensemble.RandomForestClassifier(**params)
    # initialize stratified k-fold
    kf = model_selection.StratifiedKFold(n_splits=5)
    # initialize accuracy list
    accuracies = []
    # loop over all folds
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]
        xtest = x[test_idx]
        ytest = y[test_idx]
        # fit model for current fold
        model.fit(xtrain, ytrain)
        # create predictions
        preds = model.predict(xtest)
        # calculate and append accuracy
        fold_accuracy = metrics.accuracy_score(
            ytest,
            preds
        )
        accuracies.append(fold_accuracy)
    # return negative accuracy
    return -1 * np.mean(accuracies)

if __name__ == "__main__":
    # read the training data
    df = pd.read_csv("../input/mobile_train.csv")
    # features are all columns without price_range
    # note that there is no id column in this dataset
    # here we have training features
    X = df.drop("price_range", axis=1).values
    # and the targets
    y = df.price_range.values
    # define a parameter space
    param_space = [
        # max_depth is an integer between 3 and 10
        space.Integer(3, 15, name="max_depth"),
        # n_estimators is an integer between 50 and 1500
        space.Integer(100, 1500, name="n_estimators"),
        # criterion is a category. here we define list of categories
        space.Categorical(["gini", "entropy"], name="criterion"),
        # you can also have Real numbered space and define a
        # distribution you want to pick it from
        space.Real(0.01, 1, prior="uniform", name="max_features")
    ]
    # make a list of param names
    # this has to be same order as the search space
    # inside the main function
    param_names = [
        "max_depth",
        "n_estimators",
        "criterion",
        "max_features"
    ]
    # by using functools partial, i am creating a
    # new function which has same parameters as the
    # optimize function except for the fact that
    # only one param, i.e. the "params" parameter is
    # required. this is how gp_minimize expectrs the
    # optimization function to be. you can get rid of this
    # by reading data inside the optimize function or by
    # defining the optimize function here.
    optimization_function = partial(
        optimize,
        param_names = param_names,
        x = X,
        y = y
    )
    # now we call gp_minimize from scikit-optimize
    # gp_minimize uses bayesian optimization for
    # minimization of the optimization function.
    # we need a space of parameters, the function itself,
    # the number of calls/iterations we want to have
    result = gp_minimize(
        optimization_function,
        dimensions = param_space,
        n_calls = 15,
        n_random_starts = 10,
        verbose = 10
    )
    # create best params dict and print it
    best_params = dict(
        zip(
            param_names,
            result.x
        )
    )
    print(best_params)

### Plotting convergence
from skopt.plots import plot_convergence
plot_convergence(result)

### rf_hyperopt.py
import numpy as np
import pandas as pd
from functools import partial
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

def optimize(params, x, y):
    """
    The main optimization function.
    This function takes all the arguments from the search space
    and training features and targets. It then initializes
    the models by setting the chosen parameters and runs
    cross-validation and returns a negative accuracy score
    :param params: dict of params from hyperopt
    :param x: training data
    :param y: labels/targets
    :return: negative accuracy after 5 folds
    """
    # initialize model with current parameters
    model = ensemble.RandomForestClassifier(**params)
    # initialize stratified k-fold
    kf = model_selection.StratifiedKFold(n_splits=5)
    .
    .
    .
    # return negative accuracy
    return -1 * np.mean(accuracies)

if __name__ == "__main__":
    # read the training data
    df = pd.read_csv("../input/mobile_train.csv")
    # features are all columns without price_range
    # note that there is no id column in this dataset
    # here we have training features
    X = df.drop("price_range", axis=1).values
    # and the targets
    y = df.price_range.values
    # define a parameter space
    # now we use hyperopt
    param_space = {
        # quniform gives round(uniform(low, high)/q)*q
        # we want int values for depth and estimators
        "max_depth": scope.int(hp.quniform("max_depth", 1, 15, 1)),
        "n_estimators": scope.int(
            hp.quniform("n_estimators", 100, 1500, 1)
        ),
        # choice chooses from a list of values
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
        # uniform chooses a value between two values
        "max_features": hp.uniform("max_features", 0, 1)
    }
    # partial function
    optimization_function = partial(
        optimize,
        x=X,
        y=y
    )
    # initialize trials to keep logging information
    trials = Trials()
    # run hyperopt
    hopt = fmin(
        fn=optimization_function,
        space=param_space,
        algo=tpe.suggest,
        max_evals=15,
        trials=trials
    )
    print(hopt)




#################################################################################################
# Chapter 10: Approaching Image Classification & Segmentation
#################################################################################################

### displaying random image
import numpy as np
import matplotlib.pyplot as plt
# generate random numpy array with values from 0 to 255
# and a size of 256*256
random_image = np.random.randint(0, 256, (256, 256))
# initialize plot
plt.figure(figsize=(7, 7))
# show grayscale image, nb: cmap, vmin and vmax
plt.imshow(random_image, cmap="gray", vmin=0, vmax=255)
plt.show()

### classifying pneumothorax images with Random Forest Classifier
import os
import numpy as np
import pandas as pd
from PIL import image
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from tqdm import tqdm

def create_dataset(training_df, image_dir):
    """
    This function takes the training dataframe
    and outputs training array and labels
    :param training_df: dataframe with ImageId, Target columns
    :param image_dir: location of images (folder), string
    :return: X, y (training array with features and labels)
    """
    # create empty list to store image vectors
    images = []
    # create empty list to store targets
    targets = []
    # loop over the dataframe
    for index, row in tqdm(
        training_df.iterrows(),
        total=len(training_df),
        desc="processing images"
    ):
        # get image id
        image_id = row["ImageId"]
        # create image path
        image_path = os.path.join(image_dir, image_id)
        # open image using PIL
        image = Image.open(image_path + ".png")
        # resize image to 256x256. we use bilinear resampling
        image = image.resize((256, 256), resample=Image.BILINEAR)
        # convert image to array
        image = np.array(image)
        # ravel
        image = image.ravel()
        # append images and targets lists
        images.append(image)
        targets.append(int(row["target"]))
    # convert list of list of images to numpy array
    images = np.array(images)
    # print size of this array
    print(images.shape)
    return images, targets

if __name__ == "__main__":
    csv_path = "/home/abhishek/workspace/siim_png/train.csv"
    image_path = "/home/abhishek/workspace/siim_png/train_png/"
    # read CSV with imageid and target columns
    df = pd.read_csv(csv_path)
    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1
    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    # fetch labels
    y = df.target.values
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)
    # fill the new kfold column
    for f_, (t_,v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f
    # we go over the folds created
    for fold_ in range(5):
        # temporary dataframes for train and test
        train_df = df[df.kfold != fold_].reset_index(drop=True)
        test_df = df[df.kfold == fold_].reset_index(drop=True)
        # create train dataset
        # you can move this outside to save some computation time
        xtrain, ytrain = create_dataset(train_df, image_path)
        # create test dataset
        # you can move this outside to save some computation time
        xtest, ytest = create_dataset(test_df, image_path)
        # fit random forest without any modification of params
        clf = ensemble.RandomForestClassifier(n_jobs=-1)
        clf.fit(xtrain, ytrain)
        # predict probability of class 1
        preds = clf.predict_proba(xtest)[:,1]
        # print results
        print(f"FOLD: {fold_}")
        print(f"AUC = {metric.roc_auc_score(ytest, preds)}")
        print("")

### AlexNet implementation using PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # convolution part
        self.conv1 = nn.Conv2d(
            in_channels = 3,
            out_channels = 96,
            kernel_size = 11,
            stride = 4,
            padding = 0
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size = 3,
            stride = 2
        )
        self.conv2 = nn.Conv2d(
            in_channels = 96,
            out_channels = 256,
            kernel_size = 5,
            stride = 1,
            padding = 2
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size = 3,
            stride = 2
        )
        self.conv3 = nn.Conv2d(
            in_channels = 256,
            out_channels = 384,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )
        self.conv4 = nn.Conv2d(
            in_channels = 384,
            out_channels = 256,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )
        self.pool3 = nn.MaxPool2d(
            kernel_size = 3,
            stride = 2
        )
        # dense part
        self.fc1 = nn.Linear(
            in_features = 9216,
            out_features = 4096
        )
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(
            in_features = 4096,
            out_faetures = 4096
        )
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(
            in_features = 4096,
            out_features = 1000
        )
    def forward(self, image):
        # get the batch size, channels, height and width
        # of the input batch of images
        # original size: (bs, 3, 227, 227)
        bs, c, h, w = image.size()
        x = F.relu(self.conv1(image)) # size: (bs, 96, 55, 55)
        x = self.pool1(x) # size: (bs, 96, 27, 27)
        x = F.relu(self.conv2(x)) # size: (bs, 256, 27, 27)
        x = self.pool2(x) # size: (bs, 256, 13, 13)
        x = F.relu(self.conv3(x)) # size: (bs, 384, 13, 13)
        x = F.relu(self.conv4(x)) # size: (bs, 256, 13, 13)
        x = self.pool3(x) # size: (bs, 256, 6, 6)
        x = x.view(bs, -1) # size: (bs, 9216)
        x = F.relu(self.fc1(x)) # size: (bs, 4096)
        x = self.dropout1(x) # size: (bs, 4096)
        # dropout does not change size
        # dropout is used for regularization
        # 0.3 dropout means that only 70% of the nodes
        # of the current layer are used for the next layer
        x = F.relu(self.fc2(x)) # size: (bs, 4096)
        x = self.dropout2(x) # size: (bs, 4096)
        x = F.relu(self.fc3(x)) # size: (bs, 1000)
        # 1000 is number of classes in ImageNet Dataset
        # softmax is an activation function that converts
        # linear output to probabilities that add up to 1
        # for each sample in the batch
        x = torch.softmax(x, axis=1) # size: (bs, 1000)
        return x

### creating an image classification framework
# Implement Kfold later as not given in the book but will be similar to previous chapters
### dataset.py
import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
# sometimes, you will have images without an ending bit
# this takes care of those kind of (corrupt) images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ClassificationDataset:
    """
    A general classification dataset class that you can use for all
    kinds of image classification problems. For example,
    binary classification, multi-class, multi-label classification
    """
    def __init__(
            self,
            image_paths,
            targets,
            resize=None,
            augmentations=None
    ):
        """
        :param image_paths: list of path to images
        :param targets: numpy array
        :param resize: tuple, e.g. (256, 256), resizes image if not None
        :param augmentations: albumentation augmentations
        """
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations

    def __len__(self):
        """
        Return the total number of samples in the dataset
        """
        return len(self.image_paths)

    def __getitem__(self, item):
        """
        For a given "item" index, return everything we need
        to train a given model
        """
        # use PIL to open the image
        image = Image.open(self.image_paths[item])
        # convert image to RGB, we have simgle channel images
        image = image.convert("RGB")
        # grab correct targets
        targets = self.targets[item]
        # resize if needed
        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]),
                resample = Image.BILINEAR
            )
        # convert image to numpy array
        image = np.array(image)
        # if we have albumentation augmentations
        # add them to the image
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
        # pytorch expects CHW instead of HWC
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # return tensors of image and targets
        # take a look at the types!
        # for regression tasks,
        # dtype of targets will change to torch.float
        return{
            "image": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long)
        }

### engine.py
import torch
import torch.nn as nn
from tqdm import tqdm

def train(data_loader, model, optimizer, device):
    """
    This function does training for one epoch
    :param data_loader: this is the pytorch dataloader
    :param model: pytorch model
    :param optimizer: optimizer, for e.g. adam, sgd, etc
    :param device: cuda/cpu
    """
    # put the model in train mode
    model.train()
    # go over every batch of data in data loader
    for data in data_loader:
        # remember, we have image and targets
        # in our dataset class
        inputs = data["image"]
        targets = data["targets"]
        # move inputs/targets to cuda/cpu device
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)
        # zero grad the optimizer
        optimizer.zero_grad()
        # do the forward step of model
        outputs = model(inputs)
        # calculate loss
        loss = nn.BCEWithLogitsLoss()(outputs, targets.view(-1,1))
        # backward step the loss
        loss.backward()
        # step optimizer
        optimizer.step()
        # if you have a scheduler, you either need to
        # step it here or you have to step it after
        # the epoch. here, we are not using any learning
        # rate scheduler

def evaluate(data_loader, model, device):
    """
    This function does evaluation for one epoch
    :param data_loader: this is the pytorch dataloader
    :param model: pytorch model
    :param device: cuda/cpu
    """
    # put model in evaluation mode
    model.eval()
    # init lists to store targets and outputs
    final_targets = []
    final_outputs = []
    # we use no_grad context
    with torch.no_grad():
        for data in data_loader:
            inputs = data["image"]
            targets = data["targets"]
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)
            # do the forward step to generate prediction
            output = model(inputs)
            # convert targets and outputs to lists
            targets = targets.detach().cpu().numpy().tolist()
            output = output.detach().cpu().numpy().tolist()
            # extend the original list
            final_targets.extend(targets)
            final_outputs.extend(output)
    # return final output and final targets
    return final_outputs, final_targets

### model.py
import torch.nn as nn
import pretrainedmodels

def get_model(pretrained):
    if pretrained:
        model = pretrainedmodels.__dict__["alexnet"](
            pretrained="imagenet"
        )
    else:
        model = pretrainedmodels.__dict__["alexnet"](
            pretrained=None
        )
    # print the model here to know whats going on.
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(4096),
        nn.Dropout(p=0.25),
        nn.Linear(in_features=4096, out_features=2048),
        nn.ReLU(),
        nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=2048, out_features=1)
    )
    return model

### train.py
import os
import pandas as pd
import numpy as np
import albumentations
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split
import dataset
import engine
from model import get_model

if __name__ == "__main__":
    # location of train.csv and train_png folder
    # with all the png images
    data_path = "/home/abhishek/workspace/siim_png/"
    # cuda/cpu device
    device = "cuda"
    # let's train for 10 epochs
    epochs = 10
    # load the dataframe
    df = pd.read_csv(os.path.join(data_path, "train.csv"))
    # fetch all iamge ids
    images = df.ImageId.values.tolist()
    # a list with image locations
    images = [
        ps.path.join(data_path, "train_png", i + ".png") for i in images
    ]
    # binary targets numpy array
    targets = df.target.values
    # fetch out model, we will try both pretrained
    # and non-pretrained weights
    model = get_model(pretrained=True)
    # move model to device
    model.to(device)
    # mean and std values of RGB channels for imagenet dataset
    # we use these pre-calculated values when we use weights
    # from imagenet.
    # when we do not use imagenet weights, we use the mean and
    # standard deviation values of the original dataset
    # please note that this is a separate calculation
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # albumentations is an image augmentation library
    # that allows you do to many different types of image
    # augmentations. here, i am using only normalization
    # notice always_apply=True. we always want to apply
    # normalization
    aug = albumentations.Compose(
        [
            albumentations.Normalize(
                mean, std, max_pixel_value=255.0, always_apply=True
            )
        ]
    )
    # instead of using kfold, i am using train_test_split
    # with a fixed random state
    train_images, valid_images, train_targets, valid_targets =
    train_test_split(
        images, targets, stratify=targets, random_state=42
    )
    # fetch the ClassificationDataset class
    train_dataset = dataset.ClassificationDataset(
        image_paths=train_images,
        targets=train_targets,
        resize=(227, 227),
        augmentations=aug
    )
    # torch dataloader creates batches of data
    # from classification dataset class
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=4
    )
    # same for validation data
    validation_dataset = dataset.ClassificationDataset(
        image_paths=valid_images,
        targets=valid_targets,
        resize=(227,227),
        augmentations=aug
    )
    validation_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=16, shuffle=False, num_workers=4
    )
    # simple Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    # train and print auc score for all epochs
    for epoch in range(epochs):
        engine.train(train_loader, model, optimizer, device=device)
        predictions, valid_targets = engine.evaluate(
            valid_loader, model, device=device
        )
        roc_auc = metrics.roc_auc_score(
            valid_targets, predictions
        )
        print(
            F"Epoch={epoch}, Valid ROC AUC={roc_auc}"
        )

### model.py (trying resnet18)
import torch.nn as nn
import pretrainedmodels

def get_model(pretrained):
    if pretrained:
        model = pretrainedmodels.__dict__["resnet18"](
            pretrained="imagenet"
        )
    else:
        model = pretrainedmodels.__dict__["resnet18"](
            pretrained=None
        )
    # print the model here to know whats going on
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.25),
        nn.Linear(in_features=512, out_features=2048),
        nn.ReLU(),
        nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1)
        nn.Dropout(p=0.5)
        nn.Linear(in_features=2048, out_features=1)
    )
    return model

### simple_unet.py
import torch
import torch.nn as nn
from torch.nn import functional as F

def double_conv(in_channels, out_channels):
    """
    This function applies two convolutional layers
    each followed by a ReLU activation function
    :param in_channels: number of input channels
    :param out_channels: numebr of output channels
    :return: a down-conv layer
    """
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2D(out_channels, out_channels, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv

def crop_tensor(tensor, target_tensor):
    """
    Center crops a tensor to size of a given target tensor size
    Please note that this function is applicable only to
    this implementation of unet. There are a few assumptions
    in this implementation that might not be applicable to all
    networks and all other use-cases.
    Both tensors are of shape (bs, c, h, w)
    :param tensor: a tensor that needs to be cropped
    :param target_tensor: target tensor of smaller size
    :return: cropped tensor
    """
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta//2
    return tensor[
        :,
        :,
        delta:tensor_size - delta,
        delta:tensor_size - delta
    ]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # we need only one max_pool as it is not learned
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2
        )
        self.up_conv_1 = double_conv(1024, 512)
        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2
        )
        self.up_conv_2 = double_conv(512, 256)
        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2
        )
        self.up_conv_3 = double_conv(256, 128)
        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2
        )
        self.up_conv_4 = double_conv(128, 64)
        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=2,
            kernel_size=1
        )
    def forward(self, image):
        # encoder
        x1 = self.down_conv_1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_2(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)
        # decoder
        x = self.up_trans_1(x9)
        y = crop_tensor(x7, x)
        x = self.up_conv_1(torch.cat([x,y], axis=1))
        x = self.up_trans_2(x)
        y = crop_tensor(x5, x)
        x = self.up_conv_2(torch.cat([x,y], axis=1))
        x = self.up_trans_3(x)
        y = crop_tensor(x3, x)
        x = self.up_conv_3(torch.cat([x,y], axis=1))
        x = self.up_trans_4(x)
        y = crop_tensor(x1, x)
        x = self.up_conv_4(torch.cat([x,y], axis=1))
        # output layer
        out = self.out(x)
        return out

if __name__ == "__main__":
    image = torch.rand((1,1,572,572))
    model = UNet()
    print(model(image))

### Segmentation framework
### dataset.py
import os
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
from collections import defaultdict
from torchvision import transforms
from albumentation import (
Compose,
OneOf,
RandomBrightnessContrast,
RandomGamma,
ShiftScaleRotate
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class SIIMDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            image_ids,
            transform=True,
            preprocessing_fn=None
    ):
        """
        Dataset class for segmentation problem
        :param image_ids: ids of the images, list
        :param transform: True/False, no transform in validation
        :param preprocessing_fn: a function for preprocessing image
        """
        # we create an empty dictionary to store image
        # and mask paths
        self.data = defaultdict(dict)
        # for augmentations
        self.transform = transform
        # preprocessing function to normalize
        # images
        self.preprocessing_fn = preprocessing_fn
        # albumentation augmentations
        # we have shift, scale & rotate
        # applied with 80% probability
        # and then one of gamma and brightness/contrast
        # is applied to the image
        # albumentation takes care of which augmentation
        # is applied to image and mask
        self.aug = Compose(
            [
                ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=0.8
                ),
                OneOf(
                    [
                        RandomGamma(
                            gamma_limit=(90, 110)
                        ),
                        RandomBrightnessContrast(
                            brightness_limit=0.1,
                            contrast_limit=0.1
                        )
                    ],
                    p=0.5
                )
            ]
        )
        # going over all image_ids to store
        # image and mask paths
        for imgid in image_ids:
            files = glob.glob(os.path.join(TRAIN_PATH, imgid, "*.png"))
            self.data[counter] = {
                "img_path": os.path.join(
                    TRAIN_PATH, imgid + ".png"
                ),
                "mask_path": os.path.join(
                    TRAIN_PATH, imgid + "_mask.png"
                )
            }
    def __len__(self):
        # return length of dataset
        return len(self.data)
    def __getitem__(self, item):
        # for a given item index,
        # return image and mask tensors
        # read image and mask paths
        img_path = self.data[item]["img_path"]
        mask_path = self.data[item]["mask_path"]
        # read image and convert to RGB
        img = Image.open(img_path)
        img = img.convert("RGB")
        # PIL image to numpy array
        img = np.array(img)
        # read mask image
        mask = Image.open(mask_path)
        # convert tp binary float matrix
        mask = (mask >= 1).astype("float32")
        # if this is training data, apply transforms
        if self.transform is True:
            augmented = self.aug(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        # preprocess the image using provided
        # preprocessing tensors. this is basically
        # image normalization
        img = self.preprocessing_fn(img)
        # return image and mask tensors
        return {
            "image": transforms.ToTensor()(img),
            "mask": transforms.ToTensor()(mask).float()
        }

### train.py
import os
import sys
import torch
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim
from apex import amp
from collections import OrderedDict
from sklean import model_selection
from tqdm import tqdm
from torch.optim import lr_scheduler
from dataset import SIIMDataset
# training csv file path
TRAINING_CSV = "../input/train_pneumothorax.csv"
# training and test batch sizes
TRAINING_BATCH_SIZE = 16
TEST_BATCH_SIZE = 4
# number of epochs
EPOCHS = 10
# define the encoder for U-Net
# check: https://github.com/qubvel/segmentaion_models.pytorch
# for all supported encoders
ENCODER = "resnet18"
# we use imagenet pretrained weights for the encoder
ENCODER_WEIGHTS = "imagenet"
# train on gpu
DEVICE = "cuda"

def train(dataset, data_loader, model, criterion, optimizer):
    """
    training function that trains for one epoch
    :param dataset: dataset class (SIIMDataset)
    :param data_loader: torch dataset loader
    :param model: model
    :param criterion: loss function
    :param optimizer: adam, sgd, etc.
    """
    # put the model in train mode
    model.train()
    # calculate numebr of batches
    num_batches = int(len(dataset)/data_loader.batch_size)
    # init tqdm to track progress
    tk0 = tqdm(data_loader, total=num_batches)
    # loop over all batches
    for d in tk0:
        # fetch input images and masks
        # from dataset batch
        inputs = d["image"]
        targets = d["mask"]
        # move images and masks to cpu/gpu device
        inputs = inputs.to(DEVICE, dtype=torch.float)
        targets = targets.to(DEVICE, dtype=torch.float)
        # zero grad the optimizer
        optimizer.zero_grad()
        # forward step of model
        outputs = model(inputs)
        # calculate loss
        loss = criterion(outputs, targets)
        # backward loss is calculated on a scaled loss
        # context since we are using mixed precision training
        # if you are not using mixed precision training,
        # you can use loss.backward() and delete the following
        # two lines of code
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # step the optimizer
        optimizer.step()
    # close tqdm
    tk0.close()

def evaluate(dataset, data_loader, model):
    """
    evaluation function to calculate loss on validation
    set for one epoch
    :param dataset: dataset class (SIIMDataset)
    :param data_loader: torch dataset loader
    :param model: model
    """
    # put model in eval mode
    model.eval()
    # init final_loss to 0
    final_loss = 0
    # calculate number of batches and init tqdm
    num_batches = int(len(dataset)/data_loader.batch_size)
    tk0 = tqdm(data_loader, total=num_batches)
    # we need no_grad context of torch. this save memory
    with torch.no_grad():
        for d in tk0:
            inputs = d["image"]
            targets = d["mask"]
            inputs = inputs.to(DEVICE, dtype=torch.float)
            targets = targets.to(DEVICE, dtype=torch.float)
            output = model(inputs)
            loss = criterion(output, targets)
            # add loss to final loss
            final_loss += loss
    # close tqdm
    tk0.close()
    # return average loss over all batches
    return final_loss/num_batches

if __name__ == "__main__":
    # read the training csv file
    df = pd.read_csv(TRAINING_CSV)
    # split data into training and validation
    df_train, df_valid = model_selection.train_test_split(
        df, random_state=42, test_size=0.1
    )
    # training and validation images lists/arrays
    training_images = df_train.image_id.values
    validation_images = df_valid.image_id.values
    # fetch unet model from segmentation models
    # with specified encoder architecture
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1,
        activation=None
    )
    # segmentation model provides you with a preprocessing
    # function that can be used for normalizing images
    # normalization is only applied on images and not masks
    prep_fn = smp.encoders.get_preprocessing_fn(
        ENCODER,
        ENCODER_WEIGHTS
    )
    # send model to device
    model.to(DEVICE)
    # init training dataset
    # transform is True for training data
    train_dataset = SIIMDataset(
        training_images,
        transform=True,
        preprocessing_fn=prep_fn
    )
    # wrap training dataset in torch's dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAINING_BATCH_SIZE,
        shuffle=True,
        num_workers=12
    )
    # init validation dataset
    # augmentation is disabled
    valid_dataset = SIIMDataset(
        validation_images,
        transform=False,
        preprocessing_fn=prep_fn
    )
    # wrap validation dataset in torch's dataloader
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    # NOTE: define the criterion here
    # this is left as an excercise
    # code won't work without defining this
    # criterion =

    # we will use Adam optimizer for faster convergence
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # reduce learning rate when we reach a plateau on loss
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, verbose=True
    )
    # wrap model and optimizer with NVIDIA's apex
    # this is used for mixed precision training
    # if you have a GPU that supports mixed precision,
    # this is very helpful as it will allow us to fit larger images
    # and larger batches
    model, optimizer = amp.initialize(
        model, optimizer, opt_level="O1", verbosity=0
    )
    # if we have more than one GPU, we can use both of them!
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cude.device_count()} GPUs!")
        model = nn.DataParallel(model)
    # some logging
    print(f"Training batch size: {TRAINING_BATCH_SIZE}")
    print(f"Test batch size: {TEST_BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Number of training iamges: {len(train_dataset)}")
    print(f"Number of validation iamges: {len(valid_dataset)}")
    print(f"Encoder: {ENCODER}")
    # loop over all epochs
    for epoch in range(EPOCHS):
        print(f"Training Epoch: {epoch}")
        # train for one epoch
        train(
            train_dataset,
            train_loader,
            model,
            criterion,
            optimizer
        )
        print(f"Validation Epoch: {epoch}")
        # calculate validation loss
        val_log = evaluate(
            valid_dataset,
            valid_loader,
            model
        )
        # step the scheduler
        scheduler.step(val_log["loss"])
        print("\n")

### Plant PAthology challenge 2020 Kaggle
import os
import pandas as pd
import numpy as np
import albumentations
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.model_selection import train_test_split
from wtfml.engine import Engine
from wtfml.data_loaders.image import ClassificationDataLoader

class DenseCrossEntropy(nn.Module):
    # Taken from:
    # https://www.kaggle.com/pestipeti/plant-pathology-2020-pytorch
    def __init__(self):
        super(DenseCrossEntropy, self).__init__()
    def forward(self, logits, labels):
        logits = logits.float()
        labels = labels.float()
        logprobs = F.log_softmax(logits, dim=-1)
        loss = -labels*logprobs
        loss = loss.sum(-1)
        return loss.mean()

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = torchvision.models.resnet18(pretrained=True)
        in_features = self.base_model.fc.in_features
        self.out = nn.Linear(in_features, 4)
    def forward(self, image, targets=None):
        batch_size, C, H, W = image.shape
        x = self.base_model.conv1(image)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        x = self.out(x)

        loss = None
        if targets is not None:
            loss = DenseCrossEntropy()(x, targets.type_as(x))
        return x, loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str
    )
    parser.add_argument(
        "--device", type=str
    )
    parser.add_argument(
        "--epochs", type=int
    )
    args = parser.parse_args()
    df = pd.read_csv(os.path.join(args.data_path, "train.csv"))
    images = df.image_id.values.tolist()
    images = [
        os.path.join(args.data_path, "images", i+".jpg")
        for i in images
    ]
    targets = df[["healthy", "multiple_diseases", "rust", "scab"]].values
    model = Model()
    model.to(args.device)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    aug = albumentations.Compose(
        [
            albumentations.Normalize(
                mean,
                std,
                max_pixel_value=255.0,
                always_apply=True
            )
        ]
    )
    (
        train_images, valid_images,
        train_targets, valid_targets
    ) = train_test_split(images, targets)
    train_loader = ClassificationDataLoader(
        image_paths=train_images,
        targets=train_targets,
        resize=(128, 128),
        augmentations=aug
    ).fetch(
        batch_size=16,
        num_workers=4,
        drop_last=False,
        shuffle=True,
        tpu=False
    )
    valid_loader = ClassificationDataLoader(
        image_paths=valid_images,
        targets=valid_targets,
        resize=(128, 128),
        augmentations=aug
    ).fetch(
        batch_size=16,
        num_workers=4,
        drop_last=False,
        shuffle=False,
        tpu=False
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=15, gamma=0.6
    )
    for epoch in range(args.epochs):
        train_loss = Engine.train(
            train_loader, model, optimizer, device=args.device
        )
        valid_loss = Engine.evaluate(
            valid_loader, model, device=args.device
        )
        print(
            f"{epoch}, Train Loss={train_loss}, Valid Loss={valid_loss}"
        )

$ python plant.py --data_path ../../plant_pathology --device cuda --epochs 2




#################################################################################################
# Chapter 11: Approaching Text Classification/Regression
#################################################################################################

### A simple approach towards text classification
def find_sentiment(sentence, pos, neg):
    """
    This function returns sentiment of sentence
    :param snetence: sentence, a string
    :param pos: set of positive words
    :param neg: set of negative words
    :return: returns positive, negative or neutral sentiment
    """
    # split sentence by a space
    # "this is a sentence!" becomes:
    # ["this", "is", "a", "sentence!"]
    # note that I'm splitting on all whitespaces
    # if you want to split by space use .split(" ")
    sentence = sentence.split()
    # make sentence into set
    sentence = set(sentence)
    # check number of common words with positive
    num_common_pos = len(sentence.intersection(pos))
    # check number of common words with negative
    num_common_neg = len(sentence.intersection(neg))
    # make conditions and return
    # see how return used eliminates if else
    if num_common_pos > num_common_neg:
        return "positive"
    if num_common_pos < num_common_neg:
        return "negative"
    return "neutral"

### Tokenization using NLTK
from nltk.tokenize import word_tokenize
sentence = "hi, how are you?"
sentence.split()
word_tokenize(sentence)

### Implementing CountVectorizer for bag of words using sklearn
from sklearn.feature_extraction.text import CountVectorizer
# create a corpus of sentences
corpus = [
    "hello, how are you?",
    "im getting bored at home. And you? What do you think?",
    "did you know about counts",
    "let's see if this works!",
    "YES!!!!"
]
# initialize CountVectorizer
ctv = CountVectorizer()
# fit the vectorizer on corpus
ctv.fit(corpus)
corpus_transformed = ctv.transform(corpus)
print(ctv.vocabulary_)

### Combining word tokenizer and bag of words
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
# create a corpus of sentences
corpus = [
    "hello, how are you?",
    "im getting bored at home. And you? What do you think?",
    "did you know about counts",
    "let's see if this works!",
    "YES!!!!"
]
# initialize CountVectorizer with word_tokenize from nltk
# as the tokenizer
ctv = CountVectorizer(tokenizer=word_tokenize, token_pattern=None)
# fit the vectorizer on corpus
ctv.fit(corpus)
corpus_transformed = ctv.transform(corpus)
print(ctv.vocabulary_)

### Using logistic regression to create first benchmark
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == "__main__":
    # read the training data
    df = pd.read_csv("../input/imdb.csv")
    # map positive to 1 and negative to 0
    df.sentiment = df.sentiment.apply(
        lambda x: 1 if x == "positive" else 0
    )
    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1
    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    # fetch labels
    y = df.sentiment.values
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)
    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f
    # we go over the folds created
    for fold_ in range(5):
        # temporary dataframes for train and test
        train_df = df[df.kfold != fold_].reset_index(drop=True)
        test_df = df[df.kfold == fold_].reset_index(drop=True)
        # initialize CountVectorizer with NLTK's word_tokenize
        # function as tokenizer
        count_vec = CountVectorizer(
            tokenizer=word_tokenize,
            token_pattern=None
        )
        # fit count_vec on training data reviews
        count_vec.fit(train_df.review)
        # transform training and validation data reviews
        xtrain = count_vec.transform(train_df.review)
        xtest = count_vec.transform(test_df.review)
        # initialize logistic regression model
        model = linear_model.LogisticRegression()
        # fit the model on training data reviews and sentiment
        model.fit(xtrain, train_df.sentiment)
        # make predictions on test data
        # threshold for predictions is 0.5
        preds = model.predict(xtest)
        # calculate accuracy
        accuracy = metrics.accuracy_score(test_df.sentiment, preds)
        print(f"Fold: {fold_}")
        print(f"Accuracy = {accuracy}")
        print("")

$ python ctv_logres.py

### Using multinomial naive bayes
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import naive_bayes
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
.
.
.
.
        # initialize naive bayes model
        model = naive_bayes.MultinomialNB()
        # fit the model on training data reviews and sentiment
        model.fit(xtrain, train_df.sentiment)
.
.
.

$ python ctv_nb.py

### Tfidf Vectorizer using sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
# create a corpus of sentences
corpus = [
    "hello, how are you?",
    "im getting bored at home. And you? What do you think?",
    "did you know about counts",
    "let's see if this works!",
    "YES!!!!"
]
# initialize TfidfVectorizer with word_tokenize from nltk
# as the tokenizer
tfv = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)
# fit the vectorizer on corpus
tfv.fit(corpus)
corpus_transformed = tfv.transform(corpus)
print(corpus_transformed)

### TfidfVectorizer with Logistic Regression
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
.
.
.
    # we go over the folds created
    for fold_ in range(5):
        # temporary dataframes for train and test
        train_df = df[df.kfold != fold_].reset_index(drop=True)
        test_df = df[df.kfold == fold_].reset_index(drop=True)
        # initialize TfidfVectorizer with NLTK's word_tokenize
        # function as tokenizer
        tfidf_vec = TfidfVectorizer(
            tokenizer=word_tokenize,
            token_pattern=None
        )
        # fit tfidf_vec on training data reviews
        tfidf_vec.fit(train_df.review)
        # transform training and validation data reviews
        xtrain = tfidf_vec.transform(train_df.review)
        xtest = tfidf_vec.transform(test_df.review)
        # initialize logistic regression model
        model = linear_model.LogisticRegression()
        # fit the model on training data reviews and sentiment
        model.fit(xtrain, train_df.sentiment)
        # make predictions on test data
        # threshold for predictions is 0.5
        preds = model.predict(xtest)
        # calculate accuracy
        accuracy = metrics.accuracy_score(test_df.sentiment, preds)
        print(f"Fold: {fold_}")
        print(f"Accuracy = {accuracy}")
        print("")

$ python tfv_logres.py

### n-gram using NLTK
from nltk import ngrams
from nltk.tokenize import word_tokenize
# let's see 3 grams
N = 3
# input sentence
sentence = "hi, how are you?"
# tokenized sentence
tokenized_sentence = word_tokenize(sentence)
# generate n_grams
n_grams = list(ngrams(tokenized_sentence, N))
print(n_grams)

### TfidfVectorizer with Logistic Regression and ngrams
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
.
.
.
    # we go over the folds created
    for fold_ in range(5):
        # temporary dataframes for train and test
        train_df = df[df.kfold != fold_].reset_index(drop=True)
        test_df = df[df.kfold == fold_].reset_index(drop=True)
        # initialize TfidfVectorizer with NLTK's word_tokenize
        # function as tokenizer
        tfidf_vec = TfidfVectorizer(
            tokenizer=word_tokenize,
            token_pattern=None,
            ngram_range=(1, 3)
        )
        # fit tfidf_vec on training data reviews
        tfidf_vec.fit(train_df.review)
        # transform training and validation data reviews
        xtrain = tfidf_vec.transform(train_df.review)
        xtest = tfidf_vec.transform(test_df.review)
        # initialize logistic regression model
        model = linear_model.LogisticRegression()
        # fit the model on training data reviews and sentiment
        model.fit(xtrain, train_df.sentiment)
        # make predictions on test data
        # threshold for predictions is 0.5
        preds = model.predict(xtest)
        # calculate accuracy
        accuracy = metrics.accuracy_score(test_df.sentiment, preds)
        print(f"Fold: {fold_}")
        print(f"Accuracy = {accuracy}")
        print("")

$ python tfv_logres_trigram.py

### Stemming and lemmantization
from nltk.stem import WordNetLemmantizer
from nltk.stem.snowball import SnowballStemmer
# initialize lemmantizer
lemmantizer = WordNetLemmantizer()
# initialize stemmer
stemmer = SnowballStemmer("english")
words = ["fishing", "fishes", "fished"]
for word in words:
    print(f"word={word}")
    print(f"stemmed_word={stemmer.stem(word)}")
    print(f"lemma={lemmantizer.lemmantize(word)}")
    print("")

### Applying SVD using sklearn to text
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer

# create a corpus of sentences
# we read only 10k samples from training data
# for this example
corpus = pd.read_csv("../input/imdb.csv", nrows=10000)
corpus = corpus.review.values
# initialize TfidfVectorizer with word_tokenize from nltk
# as the tokenizer
tfv = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)
# fit the vectorizer on corpus
tfv.fit(corpus)
# transform the corpus using tfidf
corpus_transformed = tfv.transform(corpus)
# initialize SVD with 10 components
svd = decomposition.TruncatedSVD(n_components=10)
# fit svd
corpus_svd = svd.fit(corpus_transformed)
# choose first sample and create a dictionary
# of feature names and their scores from svd
# you can change the sample_index variable to
# get dictionary for any other sample
sample_index = 0
feature_scores = dict(
    zip(
        tfv.get_feature_names()
        corpus_svd.components_[sample_index]
    )
)
# once we have the dictionary, we can now
# sort it in decreasing order and get the
# top N topics
N = 5
print(sorted(feature_scores, key=feature_scores.get, reverse=True)[:N])

### We can use a loop for multiple samples
N = 5
for sample_index in range(5):
    feature_scores = dict(
        zip(
            tfv.get_feature_names()
            corpus_svd.components_[sample_index]
        )
    )
    print(
        sorted(
            feature_scores,
            key=feature_scores.get,
            reverse=True
        )[:N]
    )

### cleaning the data and then feeding to SVD
import re
import string

def clean_text(s):
    """
    This function cleans the text a bit
    :param s: string
    :return: cleaned string
    """
    # split by all whitespaces
    s = s.split()
    # join tokens by single space
    # why we do this?
    # this will remove all kinds of weird space
    # "hi.    how are you?" becomes
    # "hi. how are you"
    s = " ".join(s)
    # remove all punctuations using regex and string module
    s = re.sub(f"[{re.escape(string.punctuation)}]", "", s)
    # you can add more cleaning here if you want
    # and then return the cleaned string
    return s

import pandas as pd
.
corpus = pd.read_csv("../input/imdb.csv", nrows=10000)
corpus.loc[:, "review"] = corpus.review.apply(clean_text)
.
.

### sentence vector
import numpy as np

def sentence_to_vec(s, embedding_dict, stop_words, tokenizer):
    """
    Given a sentence and other information
    this function returns embedding for the whole sentence
    :param s: sentence, string
    :param embedding_dict: dictionary word:vector
    :param stop_words: list of stop words, if any
    :param tokenizer: a tokenization function
    """
    # convert sentence to string and lowercase it
    words = str(s).lower()
    # tokenize the sentece
    words = tokenizer(words)
    # remove stop word tokens
    words = [w for w in words if w not in stop_words]
    # keep only alpha-numeric tokens
    words = [w for w in words if w.isalpha()]
    # initialize empty list to store embeddings
    M = []
    for w in words:
        # for every word, fetch the embedding from
        # the dictionary and append to list of
        # embeddings
        if w in embedding_dict:
            M.append(embedding_dict[w])
    # if we don't have any vectors, return zeros
    if len(M) == 0:
        return np.zeros(300)
    # convert list of embeddings to array
    M = np.array(M)
    # calculate sum over axis zero
    v = M.sum(axis=0)
    # return normalized vector
    return v/np.sqrt((v**2).sum())

### Using fastText to imporve the results
### fasttext.py
import io
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer

def load_vectors(fname):
    # taken from: https://fasttext.cc/docs/en/english-vectors.html
    fin = io.open(
        fname,
        'r',
        encoding='utf-8',
        newline='\n',
        errors='ignore'
    )
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data

def sentence_to_vec(s, embedding_dict, stop_words, tokenizer):
    .
    .
    .
    return 0

if __name__ == "__main__":
    # read the training data
    df = pd.read_csv("../input/imdb.csv")
    # map positive to 1 and negative to 0
    df.sentiment = df.sentiment.apply(
        lambda x: 1 if x == "positive" else 0
    )
    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    # load embeddings into memory
    print("Loading embeddings")
    embeddings = load_vectors("../input/crawl-300d-2M.vec")
    # create sentece embeddings
    print("Creating sentence vectors")
    vectors = []
    for review in df.review.values:
        vectors.append(
            sentence_to_vec(
                s = review,
                embedding_dict = embeddings,
                stop_words = [],
                tokenizer = word_tokenize
            )
        )
    vectors = np.array(vectors)
    # fetch labels
    y = df.sentiment.values
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)
    # fill the new kfold column
    for fold_, (t_, v_) in enumerate(kf.split(X=vectors, y=y)):
        print(f"Training fold: {fold_}")
        # temporary dataframes for train and test
        xtrain = vectors[t_, :]
        ytrain = y[t_]
        xtest = vectors[v_, :]
        ytest = y[v_]
        # initialize logistic regression model
        model = linear_model.LogisticRegression()
        # fit the model on training data reviews and sentiment
        model.fit(xtrain, ytrain)
        # make predictions on test data
        # threshold for predictions is 0.5
        preds = model.predict(xtest)
        # calculate accuracy
        accuracy = metrics.accuracy_score(ytest, preds)
        print(f"Accuracy = {accuracy}")
        print("")

$ python fasttext.py

### Project
### create_folds.py
import pandas as pd
from sklearn import model_selection

if __name__=="__main__":
    # read training data
    df = pd.read_csv("../input/imdb.csv")
    # map positive to 1 and negative to 0
    df.sentiment = df.sentiment.apply(
        lambda x: 1 if x == "positive" else 0
    )
    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1
    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    # fetch labels
    y = df.sentiment.values
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)
    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    # save the new csv with kfold column
    df.to_csv("../input/imdb_folds.csv", index=False)

### dataset.py
import torch

class IMDBDataset:
    def __init__(self, reviews, targets):
        """
        :param reviews: this is a numpy array
        :param targets: a vector, numpy array
        """
        self.reviews = reviews
        self.target = targets
    def __getitem__(self, item):
        # for any given item, which is an int,
        # return review and targets as torch tensor
        # item is the index of the item in concern
        review = self.reviews[item,:]
        target = self.target[item]
        return {
            "review": torch.tensor(review, dtype=torch.long)
            "target": torch.tensor(target, dtype=torch.float)
        }

### lstm.py
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, embedding_matrix):
        """
        :param embedding_matrix: numpy array with vectors for all words
        """
        super(LSTM, self).__init__()
        # number of words = number of rows in embedding matrix
        num_words = embedding_matrix.shape[0]
        # dimension of embedding is num of columns in the matrix
        embed_dim = embedding_matrix.shape[1]
        # we define an input embedding layer
        self.embedding = nn.Embedding(
            num_embeddings = num_words,
            embedding_dim = embed_dim
        )
        # embedding matrix is used as weights of
        # the embedding layer
        self.embedding.weight = nn.Parameter(
            torch.tensor(
                embedding_matrix,
                dtype=torch.float32
            )
        )
        # we don't want to train the pretrained embeddings
        self.embedding.weight.requires_grad = False
        # a simple bidirectional LSTM with
        # hidden size of 128
        self.lstm = nn.LSTM(
            embed_dim,
            128,
            bidirectional=True,
            batch_first=True
        )
        # output layer which is a linear layer
        # we have only one output
        # input (512) = 128 + 128 for mean and same for max pooling
        self.out = nn.Linear(512, 1)
    def forward(self, x):
        # pass data through embedding layer
        # the input is just the tokens
        x = self.embedding(x)
        # move embedding output to lstm
        x, _ = self.lstm(x)
        # apply mean and max pooling on lstm output
        avg_pool = torch.mean(x, 1)
        max_pool, _ = torch.max(x, 1)
        # concatenate mean and max pooling
        # this is why size is 512
        # 128 for each direction
        # avg_pool = 256 and max_pool = 256
        out = torch.cat((avg_pool, max_pool), 1)
        # pass through the output layer and return the output
        out = self.out(out)
        # return linear output
        return out

### engine.py
import torch
import torch.nn as nn

def train(data_loader, model, optimizer, device):
    """
    This is the main training function that trains model
    for one epoch
    :param data_loader: this is the torch dataloader
    :param model: model (lstm model)
    :param optimizer: torch optimizer, e.g. adam, sgd, etc.
    :param device: this can be "cuda" or "cpu"
    """
    # set model to training mode
    model.train()
    # go through batches of data in data loader
    for data in data_loader:
        # fetch review and target from the dict
        reviews = data["review"]
        targets = data["target"]
        # move the data to device that we want to use
        reviews = reviews.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        # clear the gradients
        optimizer.zero_grad()
        # make predictions from the model
        predictions = model(reviews)
        # calculate the loss
        loss = nn.BCEWithLogitsLoss()(
            predictions,
            targets.view(-1,1)
        )
        # compute gradient of loss w.r.t
        # all parameters of the model that are trainable
        loss.backward()
        # single optimization step
        optimizer.step()

def evaluate(data_loader, model, device):
    # initialize empty lists to store predictions
    # and targets
    final_predictions = []
    final_targets = []
    # put the model in eval mode
    model.eval()
    # disable gradient calculation
    with torch.no_grad():
        for data in data_loader:
            reviews = data["review"]
            targets = data["target"]
            reviews = reviews.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)
            # make predictions
            predictions = model(reviews)
            # move predictions and targets to list
            # we need to move predictions and targets to cpu too
            predictions = predictions.cpu().numpy().tolist()
            targets = data["target"].cpu().numpy().tolist()
            final_predictions.extend(predictions)
            final_targets.extend(targets)
    # return final predictions and targets
    return final_predictions, final_targets

### train.py
import io
import torch
import numpy as np
import pandas as pd
# yes, we use tensorflow
# but not for training the model!
import tensorflow as tf
from sklearn import metrics
import config
import dataset
import engine
import lstm

def load_vectors(fname):
    # take from: https://fasttext.cc/docs/en/english-vectors.html
    fin = io.open(
        fname,
        'r',
        encoding='utf-8',
        newline='\n',
        errors='ignore'
    )
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data

def create_embedding_matrix(word_index, embedding_dict):
    """
    This function creates the embedding matrix.
    :param word_index: a dictionary with word:index_value
    :param embedding_dict: a dictionary with word:embedding_vector
    :return: a numpy array with embedding vectors for all known words
    """
    # initialize matrix with zeros
    embedding_matrix = np.zeros((len(word_index) +1, 300))
    # loop over all the words
    for word, i in word_index.items():
        # if word is found in pre-trained embeddings,
        # update the matrix. if the word is not found,
        # the vector is zeros!
        if word in embedding_dict:
            embedding_matrix[i] = embedding_dict[word]
    # return embedding matrix
    return embedding_matrix

def run(df, fold):
    """
    Run training and validation for a given fold
    and dataset
    :param df: pandas dataframe with kfold column
    :param fold: current fold, int
    """
    # fetch training dataframe
    train_df = df[df.kfold != fold].reset_index(drop=True)
    # fetch validation dataframe
    valid_df = df[df.kfold == fold].reset_index(drop=True)
    print("Fitting tokenizer")
    # we use tf.keras for tokenization
    # you can use your own tokenizer and then you can
    # get rid of tensorflow
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.review.values.tolist())
    # convert training data to sequences
    # for example : "bad movie" gets converted to
    # [24, 27] where 24 is the index for bad and 27 is the
    # index for movie
    xtrain = tokenizer.texts_to_sequences(train_df.review.values)
    # similarly convert validation data to squences
    xtest = tokenizer.texts_to_sequences(valid_df.review.values)
    # zero pad the training sequences given the maximum length
    # this padding is done on left hand side
    # if sequence is > MAX_LEN, it is truncated to left hand side too
    xtrain = tf.keras.preprocessing.sequence.pad_sequences(
        xtrain, maxlen=config.MAX_LEN
    )
    # zero pad the validation sequences
    xtest = tf.kears.preprocessing.sequence.pad_sequences(
        xtest, maxlen=config.MAX_LEN
    )
    # initialize dataset class for training
    train_dataset = dataset.IMDBDataset(
        reviews=xtrain,
        targets=train_df.sentiment.values
    )
    # create torch dataloader for training
    # torch dataloader loads the data using dataset
    # class in batches specified by batch size
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=2
    )
    # initialize dataset class for validation
    valid_dataset = dataset.IMDBDataset(
        reviews=xtest,
        targets=valid_df.sentiment.values
    )
    # create torch dataloader for validation
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )
    print("Loading embeddings")
    # load embeddings as shown previously
    embedding_dict = load_vectors("../input/crawl-300d-2M.vec")
    embedding_matrix = create_embedding_matrix(
        tokenizer.word_index, embedding_dict
    )
    # create torch device, since we use gpu, we are using cuda
    device = torch.device("cuda")
    # fetch our LSTM model
    model = lstm.LSTM(embedding_matrix)
    # send model to device
    model.to(device)
    # initialize Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Training Model")
    # set best accuracy to zero
    best_accuracy = 0
    # set early stopping counter to zero
    early_stopping_counter = 0
    # train and validate for all epochs
    for epoch in range(config.EPOCHS):
        # train one epoch
        engine.train(train_data_loader, model, optimizer, device)
        # validate
        outputs, targets = engine.evaluate(
            valid_data_loader, model, device
        )
        # use threshold of 0.5
        # please note we are using linear layer and no sigmoid
        # you should do this 0.5 threshold after sigmoid
        outputs = np.array(outputs) >= 0.5
        # calculate accuracy
        accuracy = metrics.accuracy_score(targets, outputs)
        print(
            f"FOLD:{fold}, Epoch:{epoch}, Accuracy Score={accuracy}"
        )
        # simple early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        else:
            early_stopping_counter += 1
        if early_stopping_counter > 2:
            break

if __name__ == '__main__':
    # load data
    df = pd.read_csv("../input/imdb_folds.csv")
    # train for all folds
    run(df, fold=0)
    run(df, fold=1)
    run(df, fold=2)
    run(df, fold=3)
    run(df, fold=4)















































































































































































































































































































































































































































































































































































































































 
















    



































