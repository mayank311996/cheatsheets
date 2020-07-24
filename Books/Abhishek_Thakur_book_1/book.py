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





































































































































































































































































































































































































































































































































































































 
















    



































