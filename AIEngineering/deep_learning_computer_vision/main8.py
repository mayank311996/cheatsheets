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


def get_dataset(batch_size=64):
    """
    Scales and shuffles the datasets
    :param batch_size: Desired batch size
    :return: scaled and shuffled train, test and val datasets
    """
    train_dataset_scaled = fm_train.map(scale).shuffle(1900).batch(batch_size)
    test_dataset_scaled = fm_test.map(scale).batch(batch_size)
    val_dataset_scaled = fm_val.map(scale).batch(batch_size)

    return train_dataset_scaled, test_dataset_scaled, val_dataset_scaled


train_dataset, test_dataset, val_dataset = get_dataset()
train_dataset.cache()
val_dataset.cache()

feature_extractor = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/" \
                    "feature_vector/4"

feature_extractor_layer = hub.KerasLayer(
    feature_extractor,
    input_shape=(224, 224, 3)
)
feature_extractor_layer.trainable = False

model = tf.keras.Sequential([
    feature_extractor_layer,
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['acc']
)


class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.batch_losses = []
        self.batch_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()


batch_stats_callback = CollectBatchStats()

history = model.fit_generator(
    train_dataset,
    epochs=2,
    validation_data=val_dataset,
    callbacks=[batch_stats_callback]
)

plt.figure()
plt.xlabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0, 2])
plt.plot(batch_stats_callback.batch_losses)

plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0, 1])
plt.plot(batch_stats_callback.batch_acc)

result = model.evaluate(test_dataset)

for test_sample in rsp_test.take(10):
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

np.argmax(model.predict(test_dataset), axis=1)

for f0, f1 in rsp_test.map(scale).batch(400):
    y = np.argmax(f1, axis=1)
    y_pred = np.argmax(model.predict(f0), axis=1)
    print(tf.math.confusion_matrix(
        labels=y,
        predictions=y_pred,
        num_classes=3
    ))

model.save('./models/', save_format='tf')

# mobilenet is hardly around 2 MB

loaded_model = tf.kears.models.load_model('models')

for test_sample in rsp_test.take(10):
    image, label = test_sample[0], test_sample[1]
    image_scaled, label_arr = scale(test_sample[0], test_sample[1])
    image_scaled = np.expand_dims(image_scaled, axis=0)

    img = tf.keras.preprocessing.image.img_to_array(image)
    pred = loaded_model.predict(image_scaled)
    print(pred)
    plt.figure()
    plt.imshow(image)
    plt.show()
    print(f"Actual Label: {info.features['label'].names[label.numpy()]}")
    print(f"Predicted Label: {info.features['label'].names[np.argmax(pred)]}")

#########################################################################################














