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

### t-SNE visuallization of MNIST dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import datasets 
from sklearn import manifold

% matplotlib inline 
