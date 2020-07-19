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

# src/train.py
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
    






































































































































































































































































 
















    



































