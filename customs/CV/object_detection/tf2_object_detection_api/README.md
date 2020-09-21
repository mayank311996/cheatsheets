## TF2 Object Detection API

#### Installation 

> STEP 1 TensorFlow Installation 

To install TensorFlow
```bash
pip install --ignore-installed --upgrade tensorflow==2.2.0
```

Verify installation 
```bash
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

Check out GPU driver installation on https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html

> STEP 2 TF Object Detection API Installation

```bash
mkdir TensorFlow
cd TensorFlow
git clone https://github.com/tensorflow/models.git

```
