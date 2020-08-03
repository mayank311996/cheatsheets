########################################################################
# End to End Image Classification using TensorFlow 2.0 (In Colab using TPUs)
########################################################################
import tensorflow as tf
import numpy as np
import os
import datetime
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# to use tpu's data should be on local colab machine or google cloud storage
datasets, info = tfds.load(
    name='fashion_mnist',
    with_info=True,
    as_supervised=True,
    split=["train", "test"],
    batch_size=-1  # loads data into local machine RAM
)

img_train, info_train = tfds.load(name="fashion_mnist", with_info=True, split="test")
tfds.show_examples(info_train, img_train)

tf.config.experimental.list_physical_devices()

train, test = tfds.as_numpy(datasets[0]), tfds.as_numpy(datasets[1])

x_train, y_train = train[0] / 255, tf.one_hot(train[1], 10)
x_test, y_test = test[0] / 255, tf.one_hot(test[1], 10)

x_test, x_val = x_test[5000:], x_test[:5000]
y_test, y_val = y_test[5000:], y_test[:5000]

# x_train.shape
print(x_test.shape)
print(x_val.shape)


def create_model():
    """
    Creates a deep CNN model
    :return: CNN model
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, 5, padding="same", activation="relu", input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(64, 5, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(128, 5, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(256, 5, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation("softmax"))
    return model


resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="gprc://" + os.environ["COLAB_TPU_ADDR"])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)

# anything within this strategy scope will run on TPUs
with strategy.scope():
    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        # if we are using sparse_categorical_crossentropy then no need to one-hot-encode the target variable
        metrics=["accuracy"]
    )

    history = model.fit(
        x_train.astype(np.float32), np.float32(y_train),
        batch_size=256,
        epochs=25,
        steps_per_epoch=234,
        # validation_split = 0.2,
        validation_data=(x_val.astype(np.float32), np.float32(y_val)),
        validation_freq=1
    )

model.save("/tmp/fashion_tpu.hdf5")

plt.figure(figsize=(6, 8))
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train_acc", "val_acc"])
plt.show()

fmnist_load = tf.keras.models.load_model("/tmp/fashion_tpu.hdf5", compile=True)
fmnist_load.evaluate(x_test, y_test)
y_pred = fmnist_load.predict(x_test)
labels = info.features["label"].names
print(classification_report(
    y_test.numpy().argmax(axis=1),
    y_pred.argmax(axis=1),
    target_names=info.features["label"].names
))

figure = plt.figure(figsize=(20, 8))
for i, index in enumerate(np.random.choice(x_test.shape[0], size=25, replace=False)):
    ax = figure.add_subplot(5, 5, i+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_test[index]))
    predict_index = np.argmax(y_pred[index])
    true_index = np.argmax(y_test.numpy()[index])
    ax.set_title("{}({})".format(
        labels[predict_index],
        labels[true_index],
        color=("green" if predict_index == true_index else "red")
    ))














