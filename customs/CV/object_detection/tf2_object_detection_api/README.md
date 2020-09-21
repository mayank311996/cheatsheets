# TF2 Object Detection API

## Installation 

> STEP 1 TensorFlow Installation 

##### To install TensorFlow
```bash
pip install --ignore-installed --upgrade tensorflow==2.2.0
```

##### Verify installation 
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

