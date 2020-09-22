# TF2 Object Detection API

## Installation 

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

##### Cloning TF model garden
```bash
mkdir TensorFlow
cd TensorFlow
git clone https://github.com/tensorflow/models.git
```

##### Protobuf installation 

```bash
cd 
mkdir Google_Protobuf
```

Download the latest tar.gz from https://github.com/protocolbuffers/protobuf/releases
```bash
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.13.0/protobuf-all-3.13.0.tar.gz
tar -zxvf *.tar.gz
```

Adding protobuf to env variable 
```bash
gedit .bashrc
```

Add `export PATH=/home/venom/Google_Protobuf/protobuf-3.13.0/:$PATH` and close the file.
```bash
source .bashrc
```

Close and open a new terminal
Go to TensorFlow/models/research and run
```bash
protoc object_detection/protos/*.proto --python_out=.
```

>STEP 3 COCO API Installation 

```bash
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools <PATH_TO_TF>/TensorFlow/models/research/
```

>STEP 4 Install the Object Detection API

Go to TensorFlow/models/research and run
```bash
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```

>STEP 5 Test Your Installation 

Go to TensorFlow/models/research and run
```bash
python object_detection/builders/model_builder_tf2_test.py
```


## Training Custom Object Detector

>STEP 1 Preparing the Workspace

```bash
cd <PATH_TO_TensorFlow>
mkdir workspace
cd workspace
mkdir training_demo
```

```bash
cd training_demo
mkdir annotations
mkdir exported-models
mkdir images 
mkdir models
mkdir pre-trained-models
cd images
mkdir test
mkdir train
```

>STEP 2 Preparing the Dataset

##### Annotate the Dataset

Install LabelImg
```bash
pip install labelImg
```

Annotate Images 
```bash
labelImg <PATH_TO_TF>/TensorFlow/workspace/training_demo/images
```
[video](https://youtu.be/K_mFnvzyLvc)

##### Partition the Dataset

```bash
cd <PATH_TO_TensorFlow>
mkdir scripts
cd scripts 
mkdir preprocessing
```
save partition_dataset.py inside preprocessing folder.






















