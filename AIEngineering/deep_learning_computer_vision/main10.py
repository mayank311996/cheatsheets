########################################################################
# Datasets for Image Classification and Object Detection
########################################################################

import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import warnings
from sklearn.datasets import fetch_openml

#########################################################################################
warnings.filterwarnings('ignore')

datasets, info = tfds.load(
    name='malaria',
    with_info=True,
    as_supervised=True,
    split=["train"]
)

# just to visualize
train, info_train = tfds.load(
    name='malaria',
    with_info=True,
    split='train'
)
tfds.show_examples(info_train, train)

# by default dataset is in tfdata format which is good for TensorFlow
# but if you want to use this dataset in PyTorch and sklearn or
# somewhere else then you can convert it to numpy format and
# use it

datasets, info = tfds.load(
    name='fashion_mnist',
    with_info=True,
    as_supervised=True,
    split=['train', 'test'],
    batch_size=-1
)
train, test = tfds.as_numpy(datasets[0]), tfds.as_numpy(datasets[1])

faces = fetch_openml(
    name='UMIST_Faces_Cropped',
    version=1
)

print(faces.data.shape)
print(faces.target)

for i in range(6):
    idx = np.random.randint(1, 500)
    face = faces.data[idx]
    face_pixels = face.reshape(112, 92)
    plt.subplot(2, 3, i+1)
    plt.imshow(face_pixels)
    plt.axis('off')

#########################################################################################


















