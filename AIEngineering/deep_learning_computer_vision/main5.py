########################################################################
# Keras Tuner - Auto Neural Network Architecture Selection
########################################################################
# !nvidia-smi
# !pip install -U keras-tuner

import tensorflow as tf
import os
import datetime
import tensorflow_datasets as tfds
import kerastuner as kt
from kerastuner.engine.hyperparameters import HyperParameters
import matplotlib.pyplot as plt
import numpy as np

# %load_ext tensorboard

tf.config.experimental.list_physical_devices()

datasets, info = tfds.load(
    name="fashion_mnist",
    with_info=True,
    as_supervised=True,  # some datasets are availble for both supervised
    # and unsupervised
    try_gcs=True,  # for TPU based inference
    split=["train", "test"]
)

# info

print(info.features)
print(info.features["label"].num_classes)
print(info.features["label"].names)

fm_train, fm_test = datasets[0], datasets[1]
fm_val = fm_test.take(3000)
fm_test = fm_test.skip(3000)

for fm_sample in fm_train.take(5):
    image, label = fm_sample[0], fm_sample[1]
    plt.figure()
    plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap=plt.get_cmap("gray"))
    plt.show()
    print(f"Label: {label.numpy()}")
    print(f"Category: {info.features['label'].names[label.numpy()]}")


def scale(image, label):
    """
    Function to scale the image size
    :param image: image file
    :param label: target label
    :return: scaled image and label
    """
    image = tf.cast(image, tf.float32)
    image /= 255.0
    return image, label


def get_dataset(batch_size=64):
    """
    Function to apply scaled function and shuffling
    and return the data in batches
    :param batch_size: required batch size that fits in memory
    you can go up to 200 for GPUs but for CPUs keep it low like
    8 or 32 etc.
    :return: scaled and shuffled train, test and val datasets in batches
    """
    train_dataset_scaled = fm_train.map(scale).shuffle(60000).batch(batch_size)
    test_dataset_scaled = fm_test.map(scale).batch(batch_size)
    val_dataset_scaled = fm_val.map(scale).batch(batch_size)
    return train_dataset_scaled, test_dataset_scaled,


hp = HyperParameters()
hp.Choice('learning_rate', [1e-1, 1e-3])
hp.Int('conv_blocks', 3, 4, default=3)
hp.Int('hidden_size', 128, 256, step=64, default=128)


def build_model(hp):
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = inputs
    for i in range(hp.get('conv_blocks')):
        filters = hp.Int('filters_' + str(i), 32, 256, step=64)
        for _ in range(2):
            x = tf.keras.layers.Convolution2D(
                filters,
                kernel_size=(3, 3),
                padding="same"
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
        if hp.Choice('pooling_' + str(i), ["avg", "max"]) == "max":
            x = tf.keras.layers.MaxPool2D()(x)
        else:
            x = tf.keras.layers.AvgPool2D()(x)
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dense(hp.get("hidden_size"), activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.get("learning_rate")),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


tuner = kt.Hyperband(
    build_model,
    objective="val_accuracy",
    hyperparameters=hp,
    max_epochs=5,
    hyperband_iterations=2
)

tuner.search_space_summary()

train_dataset, test_dataset, val_dataset = get_dataset()
train_dataset.cache()  # caching to GPU
val_dataset.cache()

tuner.search(
    train_dataset,
    validation_data=val_dataset,
    epochs=5,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)]
)

best_model = tuner.get_best_model(1)[0]
best_model.summary()
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
best_hyperparameters.values
tuner.results_summary()
best_model.save("/tmp/fashion.hdf5")

# !nvidia-smi
# !ls -alrt /tmp/fashion.hdf5

tf.keras.backend.clear_session()
fmnist_load = tf.keras.models.load_model("/tmp/fashion.hdf5", compile=False)

for test_sample in fm_test.take(10):
    image, label = test_sample[0], test_sample[1]
    img = tf.keras.preprocessing.image.img_to_array(image)
    img = np.expand_dims(img, axis=0)
    img /= 255.
    pred = fmnist_load.predict(img)
    plt.figure()
    plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap=plt.get_cmap("gray"))
    plt.show()
    print(f"Actual label: {info.features['label'].names[label.numpy()]}")
    print(f"Predicted label: {info.features['label'].names[np.argmax(pred)]}")

_, accuracy = best_model.evaluate(test_dataset)
print(accuracy)

# remember sklearn models can also be used with keras tuner!






