import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
import tensorflow.keras.layers as L
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import efficientnet.tfkeras as efn

#########################################################################################
TEST_PATH = "/opt/ml/input/data/plant-pathology-2020-fgvc7/test.csv"
TRAIN_PATH = "/opt/ml/input/data/plant-pathology-2020-fgvc7/train.csv"
IMAGE_DIR = "/opt/ml/input/data/plant-pathology-2020-fgvc7/images/"
OUTPUT_DIR = "/opt/ml/model/"

test_data = pd.read_csv(TEST_PATH)
train_data = pd.read_csv(TRAIN_PATH)

# IMG_SIZE = (150, 150)
EPOCHS = 50
strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
AUTO = tf.data.experimental.AUTOTUNE


#########################################################################################
def format_path(st):
    return IMAGE_DIR + st + '.jpg'


def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)

    if label is None:
        return image
    else:
        return image, label


def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    if label is None:
        return image
    else:
        return image, label


def build_lrfn(lr_start=0.00001, lr_max=0.00005,
               lr_min=0.00001, lr_rampup_epochs=5,
               lr_sustain_epochs=0, lr_exp_decay=.8):
    lr_max = lr_max * strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) *\
                 lr_exp_decay**(epoch - lr_rampup_epochs
                                - lr_sustain_epochs) + lr_min
        return lr
    return lrfn


#########################################################################################
if __name__ == "__main__":
    test_paths = test_data.image_id.apply(format_path).values
    train_paths = train_data.image_id.apply(format_path).values

    train_labels = np.float32(train_data.loc[:, 'healthy':'scab'].values)
    train_paths, valid_paths, train_labels, valid_labels = \
        train_test_split(train_paths, train_labels, test_size=0.15, random_state=2020)

    train_dataset = (
        tf.data.Dataset
            .from_tensor_slices((train_paths, train_labels))
            .map(decode_image, num_parallel_calls=AUTO)
            .map(data_augment, num_parallel_calls=AUTO)
            .repeat()
            .shuffle(512)
            .batch(BATCH_SIZE)
            .prefetch(AUTO)
    )

    valid_dataset = (
        tf.data.Dataset
            .from_tensor_slices((valid_paths, valid_labels))
            .map(decode_image, num_parallel_calls=AUTO)
            .batch(BATCH_SIZE)
            .cache()
            .prefetch(AUTO)
    )

    test_dataset = (
        tf.data.Dataset
            .from_tensor_slices(test_paths)
            .map(decode_image, num_parallel_calls=AUTO)
            .batch(BATCH_SIZE)
    )

    lrfn = build_lrfn()
    STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

    with strategy.scope():
        model = tf.keras.Sequential([efn.EfficientNetB7(input_shape=(512, 512, 3),
                                                        weights='imagenet',
                                                        include_top=False),
                                     L.GlobalAveragePooling2D(),
                                     L.Dense(train_labels.shape[1],
                                             activation='softmax')])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
        # model.summary()

    history = model.fit(train_dataset,
                        epochs=EPOCHS,
                        callbacks=[lr_schedule],
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=valid_dataset)

    # test_loss, test_accuracy = model.evaluate(test_dataset)
    # print(f"Test accuracy: {test_accuracy}")

    model.save(OUTPUT_DIR + "model_EfficientNetB7.h5")
