import tensorflow as tf
import os
from google.colab import drive
import shutil
import glob
import urllib.request
import tarfile

##############################################################################
# Configs and Hyper-parameters

num_steps = 1000  # training steps
num_eval_steps = 50  # eval steps

MODELS_CONFIG = {
    'ssd_mobilenet_v2': {
        'model_name': 'ssd_mobilenet_v2_coco_2018_03_29',
        'pipeline_file': 'ssd_mobilenet_v2_coco.config',
        'batch_size': 12
    },
    'ssd_inception_v2': {
        'model_name': 'ssd_inception_v2_coco_2018_01_28',
        'pipeline_file': 'ssd_inception_v2_coco.config',
        'batch_size': 12
    },
    'faster_rcnn_inception_v2': {
        'model_name': 'faster_rcnn_inception_v2_coco_2018_01_28',
        'pipeline_file': 'faster_rcnn_inception_v2_pets.config',
        'batch_size': 12
    },
    'rfcn_resnet101': {
        'model_name': 'rfcn_resnet101_coco_2018_01_28',
        'pipeline_file': 'rfcn_resnet101_pets.config',
        'batch_size': 8
    }
}

selected_model = 'rfcn_resnet101'

MODEL = MODELS_CONFIG[selected_model]['model_name']
pipeline_file = MODELS_CONFIG[selected_model]['pipeline_file']
batch_size = MODELS_CONFIG[selected_model]['batch_size']

##############################################################################
# Install required packages

# %cd /content
# !git clone --quiet https://github.com/tensorflow/models.git
# !apt-get install -qq protobuf-compiler python-pil python-lxml python-tk
# !pip install -q Cython contextlib2 pillow lxml matplotlib
# !pip install -q pycocotools
# %cd /content/models/research
# !protoc object_detection/protos/*.proto --python_out=.
os.environ['PYTHONPATH'] += ':/content/models/research/:/content/models/' \
                            'research/slim/'
# !python object_detection/builders/model_builder_test.py

##############################################################################
# Prepare TFrecords files

# drive.mount('/content/drive')
# %cd /content/drive/My Drive/Chapter10_RCNN

# Convert train folder annotation xml files to a single csv file,
# !python xml_to_csv.py -i data/images/train -o data/annotations/
# train_labels.csv -l data/annotations

# Convert test folder annotation xml files to a single csv.
# !python xml_to_csv.py -i data/images/test -o data/annotations/
# test_labels.csv

# Generate `train.record`
# !python generate_tfrecord.py --csv_input=data/annotations/
# train_labels.csv --output_path=data/annotations/train.record
# --img_path=data/images/train --label_map data/annotations/label_map.pbtxt

# Generate `test.record`
# !python generate_tfrecord.py --csv_input=data/annotations/test_labels.csv
# --output_path=data/annotations/test.record --img_path=data/images/test
# --label_map data/annotations/label_map.pbtxt

test_record_fname = '/content/drive/My Drive/Chapter10_RCNN/' \
                    'data/annotations/test.record'
train_record_fname = '/content/drive/My Drive/Chapter10_RCNN/' \
                     'data/annotations/train.record'
label_map_pbtxt_fname = '/content/drive/My Drive/Chapter10_RCNN/' \
                        'data/annotations/label_map.pbtxt'

##############################################################################
# Download the base model

# %cd /content/models/research
MODEL_FILE = MODEL + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
DEST_DIR = '/content/models/research/pretrained_model'

if not (os.path.exists(MODEL_FILE)):
    urllib.request.urlretrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

tar = tarfile.open(MODEL_FILE)
tar.extractall()
tar.close()

os.remove(MODEL_FILE)
if os.path.exists(DEST_DIR):
    shutil.rmtree(DEST_DIR)
os.rename(MODEL, DEST_DIR)

# !echo {DEST_DIR}
# !ls -alh {DEST_DIR}

fine_tune_checkpoint = os.path.join(DEST_DIR, "model.ckpt")

##############################################################################




















