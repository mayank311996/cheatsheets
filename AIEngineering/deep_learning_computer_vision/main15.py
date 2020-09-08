########################################################################
# Tensorflow Custom Graphs with TPU - Part 2
########################################################################

import os
import tensorflow as tf
# %load_ext tensorboard

#########################################################################################
tf.config.experimental.list_physical_devices()
tf.debugging.set_log_device_placement(True)
tpu_add = 'grpc://'+os.environ['COLAB_TPU_ADDR']

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_add)
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.experimental.TPUStrategy(resolver)

# Running on CPU
# create some tensors
a_cpu = tf.constant(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]
)
b_cpu = tf.constant(
    [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ]
)
c_cpu = tf.matmul(a_cpu, b_cpu)

session = tf.Session()
print(session.run(c_cpu))

# Running on TPU
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# create some tensors
a_tpu = tf.constant(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]
)
b_tpu = tf.constant(
    [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ]
)
tpu_ops = tf.contrib.tpu.rewrite(tf.matmul, [a_tpu, b_tpu])

session = tf.Session(tpu_add)
print(session.run(tpu_ops))


#########################################################################################
# taking sparkml logistic regression model and deploying on a TPU












