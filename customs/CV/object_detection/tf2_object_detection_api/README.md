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
save `partition_dataset.py` (https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_downloads/d0e545609c5f7f49f39abc7b6a38cec3/partition_dataset.py) inside `preprocessing` folder

```bash
cd preprocessing 
python partition_dataset.py -x -i [PATH_TO_IMAGES_FOLDER] -r 0.1

# For example
# python partition_dataset.py -x -i C:/Users/sglvladi/Documents/Tensorflow/workspace/training_demo/images -r 0.1
```

##### Create Label Map

save `label_map.pbtxt` inside `training_demo/annotations` folder 

##### Create TensorFlow Records

Convert `*.xml` to `*.record`

save `generate_tfrecord.py` (https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_downloads/da4babe668a8afb093cc7776d7e630f3/generate_tfrecord.py)
inside `TensorFlow/scripts/preprocessing` folder

```bash
cd preprocessing
python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/train -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/train.record
python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/test -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/test.record

# For example
# python generate_tfrecord.py -x C:/Users/sglvladi/Documents/Tensorflow/workspace/training_demo/images/train -l C:/Users/sglvladi/Documents/Tensorflow/workspace/training_demo/annotations/label_map.pbtxt -o C:/Users/sglvladi/Documents/Tensorflow/workspace/training_demo/annotations/train.record
# python generate_tfrecord.py -x C:/Users/sglvladi/Documents/Tensorflow/workspace/training_demo/images/test -l C:/Users/sglvladi/Documents/Tensorflow2/workspace/training_demo/annotations/label_map.pbtxt -o C:/Users/sglvladi/Documents/Tensorflow/workspace/training_demo/annotations/test.record
```

>STEP 3 Configuring a Training Job

To train entirely new model (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md)

Model Zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

##### Download Pre-Trained Model

Download [SSD ResNet50 V1 FPN 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz)
and extract inside `training_demo/pre-trained-models` folder

Download [EfficientDet D1 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz)
and extract inside `training_demo/pre-trained-models` folder

##### Configuring the Training Pipeline

```bash
cd training_demo/models
mkdir my_ssd_resnet50_v1_fpn
```

Copy `training_demo/pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/pipeline.config`
inside `training_demo/models/my_ssd_resnet50_v1_fpn` directory

Change `pipeline.config` according to [doc](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html)

>STEP 4 Training the Model

Copy `TensorFlow/models/research/object_detection/model_main_tf2.py`
inside `training_demo` folder

```bash
cd training_demo
python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config
```

>STEP 5 Evaluating the Model (Optional)

Supported [matrices](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/evaluation_protocols.md)

The training process will occasionally create checkpoint files inside the `training_demo/training`
folder 

As we already installed COCO API, we just need to modify lines 
178-179 in `pipeline.config` script 

To actually run the evaluation
```bash
cd training_demo
python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config --checkpoint_dir=models/my_ssd_resnet50_v1_fpn
```

Evaluation process will take the latest `models/my_ssd_resnet50_v1_fpn/ckpt-*` 
checkpoint to evaluate the performance of the model

The results are stored in the form of tf event files `events.out.tfevents.*`
inside `models/my_ssd_resnet50_v1_fpn/eval_0`

>STEP 6 Monitor Training Job Progress using TensorBoard

```bash
cd training_demo
tensorboard --logdir=models/my_ssd_resnet50_v1_fpn
```

Go to `http://localhost:6006/` to access TensorBoard

>STEP 7 Exporting a Trained Model

After training, we need to extract newly trained inference graph, which will
be later used to perform the object detection

Copy `TensorFlow/models/research/object_detection/exporter_main_v2.py`
into `training_demo` folder

```bash
cd training_demo
python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\my_efficientdet_d1\pipeline.config --trained_checkpoint_dir .\models\my_efficientdet_d1\ --output_directory .\exported-models\my_model
```












