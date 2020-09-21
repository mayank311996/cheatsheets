import tensorflow as tf
import os
from google.colab import drive
import shutil
import glob
import urllib.request
import tarfile
import re

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
# Configuring a training pipeline

pipeline_fname = os.path.join(
    '/content/models/research/object_detection/samples/configs/',
    pipeline_file
)

assert os.path.isfile(pipeline_fname), '`{}` not exist'.format(pipeline_fname)


def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())


num_classes = get_num_classes(label_map_pbtxt_fname)
with open(pipeline_fname) as f:
    s = f.read()
with open(pipeline_fname, 'w') as f:
    # fine_tune_checkpoint
    s = re.sub('fine_tune_checkpoint: ".*?"',
               'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)

    # tfrecord files train and test.
    s = re.sub(
        '(input_path: ".*?)(train.record)(.*?")',
        'input_path: "{}"'.format(train_record_fname), s)
    s = re.sub(
        '(input_path: ".*?)(val.record)(.*?")',
        'input_path: "{}"'.format(test_record_fname), s)

    # label_map_path
    s = re.sub(
        'label_map_path: ".*?"',
        'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

    # Set training batch_size.
    s = re.sub('batch_size: [0-9]+',
               'batch_size: {}'.format(batch_size), s)

    # Set training steps, num_steps
    s = re.sub('num_steps: [0-9]+',
               'num_steps: {}'.format(num_steps), s)

    # Set number of classes num_classes.
    s = re.sub('num_classes: [0-9]+',
               'num_classes: {}'.format(num_classes), s)
    f.write(s)

# !cat {pipeline_fname}
model_dir = 'training/'
# Optionally remove content in output model directory to fresh start.
# !rm -rf {model_dir}
os.makedirs(model_dir, exist_ok=True)

##############################################################################
# TensorBoard running on a local machine

# tensorboard_callback = tf.keras.callbacks.TensorBoard(
#     log_dir=log_dir,
#     histogram_freq=1
# )
# history = model.fit(
#     x=x_train,
#     y=y_train,
#     epochs=25,
#     validation_data=(x_test, y_test),
#     callbacks=[tensorboard_callback]
# )

# %tensorboard --logdir logs/fit

##############################################################################
# TensorBoard running on Google Colab

# !wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
# !unzip -o ngrok-stable-linux-amd64.zip

LOG_DIR = model_dir
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)

get_ipython().system_raw('./ngrok http 6006 &')

# ! curl -s http://localhost:4040/api/tunnels | python3 -c \
#     "import sys, json;
#     print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

##############################################################################


















