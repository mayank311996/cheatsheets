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
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    logdir,
    histogram_freq=1
)

train_dataset, test_dataset, val_dataset = get_dataset()
train_dataset.cache()
val_dataset.cache()

q_aware_model.fit(
    train_dataset,
    epochs=5,
    validation_data=val_dataset,
    callbacks=[tensorboard_callback]
)

model.save("/tmp/fashion.hdf5")

# %tensorboard --logdir /tmp/logs
# the size of the train model will be small and you still need
# to convert it to work with int numbers

# The quantization technique is not only useful for edge devices
# but can also be used when low inference time is desired

q_aware_model.evaluate(test_dataset, verbose=0)

converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# over here we are choosing default optimization
# however, you can choose to optimize for specific parameter
# like memory or size etc.

quantized_tflite_model = converter.convert()

quantized_model_size = len(quantized_tflite_model)/1024
print(f"Quantized model size = {quantized_model_size}KBs")

interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
interpreter.allocate_tensors()

input_tensor_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.tensor(interpreter.get_output_details()[0]["index"])

interpreter.get_tensor_details()

prediction_output = []
accurate_count = 0

for test_image in fm_test.map(scale):
    test_image_p = np.expand_dims(test_image[0], axis=0).astype(np.float32)
    interpreter.set_tensor(input_tensor_index, test_image_p)

    interpreter.invoke()
    out = np.argmax(output_index()[0])
    prediction_output.append(out)

    if out == test_image[1].numpy():
        accurate_count += 1

accuracy = accurate_count/len(prediction_output)
print(accuracy)

#########################################################################################







