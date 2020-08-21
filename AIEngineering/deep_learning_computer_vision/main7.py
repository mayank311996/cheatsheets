########################################################################
# Deep Learning with Tensorflow - Quantization Aware Training
########################################################################

# Two types of quantization: post model and quantization aware training
# the first one we discussed in main6.py which optimizes once model
# is trained where as in second case you do optimize during training.

# !pip install -q tensorflow-model-optimization
import tensoflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.config.experimental.list_physical_devices()

datasets, info = tfds.load(
    name="fashion_mnist",
    with_info=True,
    as_supervised=True,
    try_gcs=True,
    split=["train", "test"]
)
print(info)
print(info.features)
print(info.features["label"].num_classes)
print(info.features["label"].names)

fm_train, fm_test = datasets[0], datasets[1]
fm_val = fm_test.take(300)
fm_test = fm_test.skip(300)

print(fm_train)
len(list(fm_train))

for fm_sample in fm_train.take(5):
    image, label = fm_sample[0], fm_sample[1]
    plt.figure()
    plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap=plt.get_cmap("gray"))
    plt.show()
    print(f"Label: {label.numpy()}")
    print(f"Category: {info.features['label'].names[label.numpy()]}")


def scale(image, label):
    """
    Scales the input image by dividing with 255.0
    :param image: input image
    :param label: label of the image
    :return: scaled image with label
    """
    image = tf.cast(image, tf.float32)
    image /= 255.0

    return image, label









