########################################################################
# Machine Learning Model Deployment on Edge Devices - Part 1
########################################################################

import time
import tensorflow as tf
from google.colab import drive
drive.mount('/content/drive')

model = tf.keras.models.load_mdoel('/content/drive/My drive/'
                                   'colab notebooks/models/fashion_tpu.hdf5')
print(model.summary())

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# it efficiently converts the model into smaller size without
# compromising on the accuracy

tflite_model = converter.convert()

model_size = len(tflite_model)/1024
print(f"mdoel size = {model_size}KBs.")

# This changes types of inputs and outputs from float to int and so
# this can impact the accuracy but not much
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.
# TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
# converter.representative_dataset = dataset_gen
tflite_quantized_model = converter.convert()

quantized_model_size = len(tflite_quantized_model)/1024
print(f"Quantized model size = {quantized_model_size}KBs.")

(x_train, y_train), (x_test, y_test) = \
    tf.keras.datasets.fashion_mnist.load_data()

print(x_test.shape)
x_test = np.expand_dims(x_test, -1)  # expanding dims to include
# batch size

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_tensor_index = interpreter.get_input_details()[0]["index"]
output = interpreter.tensor(interpreter.get_output_details()[0]["index"])

interpreter.get_tensor_details()

all_layers = interpreter.get_tensor_details()
for layer in all_layers:
    print(interpreter.get_tensor(layer["index"]))

first_layer = model.layers[0]
weights = first_layer.get_weights()[0]
print(weights)

prediction_output = []
for test_image in x_test:

    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_tensor_index, test_image)

    interpreter.invoke()

    out = np.argmax(output()[0])
    prediction_output.append(out)

accurate_count = 0
for index in range(len(prediction_output)):
    if prediction_output[index] == y_test[index]:
        accurate_count += 1
accuracy = accurate_count*1.0/len(prediction_output)
print(accuracy)

index = 0
for test_image in x_test[:10]:
    img = np.expand_dims(test_image, axis=0)
    tf_prediction = model.predict(img)
    plt.figure()
    plt.imshow(img[0, :, :, 0].astype(np.float32), cmap=plt.get_cmap("gray"))
    plt.show()
    print(f"Actual label: {y_test[index]}")
    print(f"Tensorflow Regular Model: {np.argmax(tf_prediction)}")
    print(f"Tensorflow Quantization Model: {prediction_output[index]}")
    index = index + 1

start_time = time.time()
for test_image in x_test:
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_tensor_index, test_image)

    interpreter.invoke()
print(f"Total Quantization Predict Time: {str(time.time()-start_time)} seconds")

start_time = time.time()
for test_image in x_test:
    img = np.expand_dims(test_image, axis=0)
    model.predict(img)
print(f"Total Regular Predict Time: {str(time.time()-start_time)} seconds")





















