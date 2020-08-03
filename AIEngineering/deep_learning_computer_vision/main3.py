########################################################################
# Transfer Learning - Image Classification using TensorFlow
########################################################################
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub  # contains pretrained computer vision and nlp models
import os
import numpy as np
import tensorflow_datasets as tfds
import warnings

warnings.filterwarnings("ignore")

datasets, info = tfds.load(
    name="beans",
    with_info=True,
    as_supervised=True,
    split=["train", "test", "validation"]
)

# just to visualize
test, info_test = tfds.load(
    name="beans",
    with_info=True,
    split="test"
)
tfds.show_examples(info_test, test)


def scale(image, label):
    """
    Function to scale the image size
    :param image: image file
    :param label: target label
    :return: scaled image and label
    """
    image = tf.cast(image, tf.float32)
    image /= 255.0
    return tf.image.resize(image, [224, 224]), tf.one_hot(label, 3)


def get_dataset(batch_size=32):
    """
    Function to apply scaled function and shuffling
    and return the data in batches
    :param batch_size: required batch size that fits in memory
    you can go up to 200 for GPUs but for CPUs keep it low like
    8 or 32 etc.
    :return: scaled and shuffled train, test and val datasets in batches
    """
    train_dataset_scaled = datasets[0].map(scale).shuffle(1000).batch(batch_size)
    test_dataset_scaled = datasets[1].map(scale).batch(batch_size)
    val_dataset_scaled = datasets[2].map(scale).batch(batch_size)
    return train_dataset_scaled, test_dataset_scaled, val_dataset_scaled


train_dataset, test_dataset, val_dataset = get_dataset()
train_dataset.cache()  # caching to GPU
val_dataset.cache()

len(list(datasets[0]))

feature_extractor = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor_layer = hub.KerasLayer(feature_extractor, input_shape=(224, 224, 3))
feature_extractor_layer.trainable = False

model = tf.keras.Sequential([
    feature_extractor_layer,
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation="softmax")
])
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["acc"]
)

history = model.fit(train_dataset, epochs=6, validation_data=val_dataset)

result = model.evaluate(test_dataset)

for test_sample in datasets[1].take(10):
    image, label = test_sample[0], test_sample[1]
    image_scaled, label_arr = scale(test_sample[0], test_sample[1])
    image_scaled = np.expand_dims(image_scaled, axis=0)

    img = tf.keras.preprocessing.image.img_to_array(image)
    pred = model.predict(image_scaled)
    print(pred)
    plt.figure()
    plt.imshow(image)
    plt.show()
    print(f"Actual Label: {info.features['label'].names[label.numpy()]}")
    print(f"Predicted Label: {info.features['label'].names[np.argmax(pred)]}")

for f0, f1 in datasets[1].map(scale).batch(200):
    y = np.argmax(f1, axis=1)
    y_pred = np.argmax(model.predict(f0), axis=1)
    print(tf.math.confusion_matrix(labels=y, predictions=y_pred, num_classes=3))

















