import tensorflow as tf
import os
from google.colab import drive
from google.colab import files
import shutil
import glob
import urllib.request
import tarfile
import re
import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import zipfile
from collections import defaultdict
from io import StringIO
import matplotlib.pyplot as plt
from PIL import Image
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

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
# Training the model

# !python /content/models/research/object_detection/model_main.py \
#     --pipeline_config_path={pipeline_fname} \
#     --model_dir={model_dir} \
#     --alsologtostderr \
#     --num_train_steps={num_steps} \
#     --num_eval_steps={num_eval_steps}

##############################################################################
# Exporting a trained inference graph

output_directory = './fine_tuned_model'

lst = os.listdir(model_dir)
lst = [l for l in lst if 'model.ckpt-' in l and '.meta' in l]
steps=np.array([int(re.findall('\d+', l)[0]) for l in lst])
last_model = lst[steps.argmax()].replace('.meta', '')

last_model_path = os.path.join(model_dir, last_model)
print(last_model_path)

# !python /content/models/research/object_detection/
# export_inference_graph.py \
#     --input_type=image_tensor \
#     --pipeline_config_path={pipeline_fname} \
#     --output_directory={output_directory} \
#     --trained_checkpoint_prefix={last_model_path}

##############################################################################
# Inference

# !ls {output_directory}
pb_fname = os.path.join(os.path.abspath(output_directory),
                        "frozen_inference_graph.pb")
assert os.path.isfile(pb_fname), '`{}` not exist'.format(pb_fname)
# !ls -alh {pb_fname}

files.download(label_map_pbtxt_fname)

# Path to frozen detection graph. This is the actual model that is
# used for the object detection.
PATH_TO_CKPT = pb_fname

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = label_map_pbtxt_fname

# If you want to test the code with your images, just add images files
# to the PATH_TO_TEST_IMAGES_DIR.
# PATH_TO_TEST_IMAGES_DIR =  os.path.join(repo_dir_path, "test")
PATH_TO_TEST_IMAGES_DIR = '/content/drive/My Drive/Chapter10_RCNN/' \
                          'data/images/val'
assert os.path.isfile(pb_fname)
assert os.path.isfile(PATH_TO_LABELS)
TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, "*.*"))
assert len(TEST_IMAGE_PATHS) > 0, 'No image found in `{}`.'.\
    format(PATH_TO_TEST_IMAGES_DIR)
print(TEST_IMAGE_PATHS)

# %cd /content/models/research/object_detection
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
# %matplotlib inline

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=num_classes, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().\
                        get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates
                # to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.\
                    reframe_box_masks_to_image_masks(
                        detection_masks,
                        detection_boxes,
                        image.shape[0],
                        image.shape[1]
                    )
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().\
                get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor:
                                              np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as
            # appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = \
                output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = \
                    output_dict['detection_masks'][0]
    return output_dict


for image_path in TEST_IMAGE_PATHS:
    image = Image.open(image_path)
    # the array based representation of the image will be used later in
    # order to prepare the result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1,
    # None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)

##############################################################################







