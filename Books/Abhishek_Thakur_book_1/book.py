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





































