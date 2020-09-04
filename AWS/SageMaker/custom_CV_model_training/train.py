import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = "/opt/ml/input/data/flowers/"
OUTPUT_DIR = "/opt/ml/model/"

IMG_SIZE = (150, 150)
BATCH_SIZE = 128


def train():
    image_gen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2
    )

    train_data_gen = image_gen.flow_from_directory(
        batch_size=BATCH_SIZE,
        directory=DATA_DIR,
        subset="training",
        shuffle=True,
        target_size=IMG_SIZE,
        class_mode='sparse'
    )

    test_data_gen = image_gen.flow_from_directory(
        batch_size=BATCH_SIZE,
        directory=DATA_DIR,
        subset="validation",
        target_size=IMG_SIZE,
        class_mode='sparse'
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(6)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    model.fit(train_data_gen, epochs=2)
    test_loss, test_accuracy = model.evaluate(test_data_gen)
    print(f"Test accuracy: {test_accuracy}")

    model.save(OUTPUT_DIR + "model.h5")


if __name__ == "__main__":
    train()
