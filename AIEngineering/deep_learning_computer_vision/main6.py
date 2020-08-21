########################################################################
# Machine Learning Model Deployment on Edge Devices - Part 1
########################################################################

from google.colab import drive
drive.mount('/content/drive')

model = tf.keras.models.load_mdoel('/content/drive/My drive/colab notebooks/models/fashion_tpu.hdf5')
print(model.summary())

converter = tf.lite.TFLiteConverter.from_keras_model(model)

