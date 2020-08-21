########################################################################
# Deep Learning with Tensorflow - Quantization Aware Training
########################################################################

# Two types of quantization: post model and quantization aware training
# the first one we discussed in main6.py which optimizes once model
# is trained where as in second case you do optimize during training.

# !pip install -q tensorflow-model-optimization
import tensorflow_model_optimization as tfmot
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

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


def create_model():
    """
    Creates a simple keras model
    :return: keras model
    """
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            64,
            2,
            padding="same",
            activation="relu",
            input_shape=(28, 28, 1)
        )
    )
    model.add(
        tf.keras.layers.MaxPooling2D()
    )
    model.add(
        tf.keras.layers.Dropout(0.3)
    )
    model.add(
        tf.keras.layers.Conv2D(
            128,
            2,
            padding="same",
            activation="relu"
        )
    )
    model.add(
        tf.keras.layers.MaxPooling2D()
    )
    model.add(
        tf.keras.layers.Dropout(0.3)
    )
    model.add(
        tf.keras.layers.Flatten()
    )
    model.add(
        tf.keras.layers.Dense(256)
    )
    model.add(
        tf.keras.layers.Activation("relu")
    )
    model.add(
        tf.keras.layers.Dense(10)
    )
    model.add(
        tf.keras.layers.Activation("softmax")
    )
    return model


# quantization stuff is need to be taken care during training only
# otherwise model definition and other stuff is similar to normal
# training

model = create_model()
quantize_model = tfmot.quantization.keras.quantize_model
q_aware_model = quantize_model(model)
q_aware_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
q_aware_model.summary()

logdir = os.path.join("/tmp/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))











