########################################################################
# Object Detection Model on Custom Dataset
########################################################################
# !nvidia-smi
# !pip install mxnet-cu101
# !pip install autogluon
# !pip install -U ipykerenel

import os
import autogluon as ag
from autogluon import ObejctDetection as task
from autogluon import Detector
from Ipython.display import Image

root = "/tmp"
# Pascal VOC type format for data
filename_zip = ag.download(
    "https:/raw.githubusercontent.com/srivatsan88/YouTubeLI/master/dataset/nfpa.zip",
    path=root
)
filename = ag.unzip(filename_zip, root=root)

# !ls /tmp/nfpa

data_root = os.path.join(root, filename)
dataset_train = task.Dataset(data_root, classes=("nfpa",))

Image(os.path.join(data_root, "JPEGImages/pos-101.jpg"))
Image(os.path.join(data_root, "JPEGImages/pos-201.jpg"))

TIME_LIMITS = 20
EPOCHS = 25
# by default it uses YOLO3, we can change that through a parameter
detector = task.fit(
    dataset_train,
    epochs=EPOCHS,
    lr=ag.Categorical(5e-4, 1e-4),
    ngpus_per_trial=1,
    time_limits=TIME_LIMITS
)

dataset_test = task.Dataset(data_root, index_file_name="test", classes=("nfpa",))

test_map = detector.evaluate(dataset_test)
print(f"mAP on test dataset: {test_map[1][1]}")

image = "pos-230.jpg"
image_path = os.path.join(data_root, "JPEGImages", image)
ind, prob, loc = detector.predict(image_path)

image = "pos-23.jpg"
image_path = os.path.join(data_root, "JPEGImages", image)
ind, prob, loc = detector.predict(image_path)

image = "pos-201.jpg"
image_path = os.path.join(data_root, "JPEGImages", image)
ind, prob, loc = detector.predict(image_path)

savefile = "model.pkl"
detector.save(savefile)

new_detector = Detector.load(savefile)


















