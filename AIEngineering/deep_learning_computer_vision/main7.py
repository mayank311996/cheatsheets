########################################################################
# Deep Learning with Tensorflow - Quantization Aware Training
########################################################################

# Two types of quantization: post model and quantization aware training
# the first one we discussed in main6.py which optimizes once model
# is trained where as in second case you do optimize during training.

import tensoflow_datasets as tfds
import tensorflow as tf

tf.config.experimental.list_physical_devices()
