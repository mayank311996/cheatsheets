########################################################################
# Image Classifier using Transfer Learning with Tensorflow
########################################################################

import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import os
import numpy as np
import tensorflow_datasets as tfds

#########################################################################################
datasets, info = tfds.load(
    name='rock_paper_scissors',
    with_info=True,
    as_supervised=True,
    split=['train', 'test']
)
print(info)

# this is just for visualization, not used later
train, info_train = tfds.load(
    name='rock_paper_scissors',
    with_info=True,
    split='test'
)
tfds.show_examples(info_train, train)

dataset = datasets[0].concatenate(datasets[1])
dataset = dataset.shuffle(3000)

rsp_val = dataset.take(600)
rsp_test_temp = dataset.skip(600)
rsp_test = rsp_test_temp.take(400)
rsp_train = rsp_test_temp.skip(400)

len(list(rsp_train))


def scale(image, label):
    """
    Scales the input image by dividing with 255.0
    :param image: input image
    :param label: label of the image
    :return: scaled image resized with label one-hot-encoded
    """
    image = tf.cast(image, tf.float32)
    image /= 255.0

    return tf.image.resize(image, [224, 224]), tf.one_hot(label, 3)


def get_dataset(batch_size=256):
    """
    Scales and shuffles the datasets
    :param batch_size: Desired batch size
    :return: scaled and shuffled train, test and val datasets
    """
    train_dataset_scaled = fm_train.map(scale).shuffle(60000).batch(batch_size)
    test_dataset_scaled = fm_test.map(scale).batch(batch_size)
    val_dataset_scaled = fm_val.map(scale).batch(batch_size)

    return train_dataset_scaled, test_dataset_scaled, val_dataset_scaled


