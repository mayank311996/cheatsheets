import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, \
    GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pathlib

##############################################################################
img_width, img_height = 299, 299
AUTOTUNE = tf.data.experimental.AUTOTUNE
NUM_EPOCHS = 5
batchsize = 10
num_train_images = 1350
num_val_images = 156

base_model = InceptionV3(
    weigths='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)
)

train_dir = '../train'
val_dir = '../validation'
train_dir = pathlib.Path(train_dir)
val_dir = pathlib.Path(val_dir)
output_dir = '../output'

train_ds = tf.data.Dataset.list_files(str(train_dir/'*/*'))
val_ds = tf.data.Dataset.list_files(str(val_dir/'*/*'))
class_list = ["bed", "chair", "sofa"]

for f in train_ds.take(5):
    print(f.numpy())


##############################################################################
def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == class_list


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [img_width, img_height])


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


##############################################################################
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_labeled_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
# I think this will dump entire data to the memory what if you have
# really big data set?

for image, label in train_labeled_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())

val_labeled_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in val_labeled_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(batchsize)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


train_ds = prepare_for_training(train_labeled_ds)

train_image_batch, label_batch = next(iter(train_ds))

val_ds = prepare_for_training(val_labeled_ds)

val_image_batch, label_batch = next(iter(val_ds))


##############################################################################
def build_final_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    x = Flatten()(x)

    # Fine-tune from this layer onwards
    layer_adjust = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:layer_adjust]:
        layer.trainable = False

    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x)

    final_model = Model(inputs=base_model.input, outputs=predictions)

    return final_model


class_list = ["bed", "chair", "sofa"]
FC_LAYERS = [1024, 1024]
dropout = 0.3

final_model = build_final_model(base_model,
                                dropout=dropout,
                                fc_layers=FC_LAYERS,
                                num_classes=len(class_list))

outputpath=output_dir+"/model-{epoch:02d}-{val_accuracy:.2f}.hdf5"

checkpoint_callback = ModelCheckpoint(
    outputpath,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=False,
    save_weights_only=False,
    save_frequency=1
)

adam = Adam(lr=0.00001)
final_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])
history = final_model.fit(
    train_ds,
    epochs=NUM_EPOCHS,
    steps_per_epoch=num_train_images // batchsize,
    callbacks=[checkpoint_callback],
    validation_data=val_ds,
    validation_steps=num_val_images // batchsize
)

##############################################################################






