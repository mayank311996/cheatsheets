########################################################################
# End to End TensorFlow - Hands On (In Colab)
########################################################################
import tensorflow as tf
import os
import datetime
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# %load_ext tensorboard

# Both not needed for actual app
tf.debugging.set_log_device_placement(True)
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

print(fm_train)
print(list(fm_train))

for fm_sample in fm_train.take(5):
    image, label = fm_sample[0], fm_sample[1]
    plt.figure()
    plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap=plt.get_cmap("gray"))
    plt.show()
    print(f"Label: {label.numpy()}")
    print(f"Category: {info.features['label'].names[label.numpy()]}")


# !mkdir /tmp/logs
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
    return train_dataset_scaled, test_dataset_scaled, val_dataset_scaled


def create_model():
    """
    Creates a deep learning model
    :return: A CNN model
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax")
        ]
    )


model = create_model()
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["sparse_categorical_accuracy"]
)

logdir = os.path.join("/tmp/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

train_dataset, test_dataset, val_dataset = get_dataset()
train_dataset.cache()
val_dataset.cache()

model.fit(
    train_dataset,
    epochs=5,
    steps_per_epoch=20,
    validation_data=val_dataset,
    callbacks=[tensorboard_callback]
)

print(tf.distribute.get_strategy())














