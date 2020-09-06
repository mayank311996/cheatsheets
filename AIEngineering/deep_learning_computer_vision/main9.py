########################################################################
# Object Detection using YOLO - Getting Started
########################################################################

# nvidia-smi
# nvcc -V
# ls /content/drive/'My Drive'/cudnn*
# cd /usr/local/
# tar -xzvf "/content/drive/My Drive/cudnn-10.1-linux-x64-v7.6.4.38
# .solitairetheme8"
# cd /usr/local/cuda
# chmod a+r /usr/local/cuda/include/cudnn.h
# cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
# cd ~
# git clone https://github.com/AlexeyAB/darknet
# ls
# cd darknet
# cat Makefile
# sed -i 's/GPU=0/GPU=1/' Makefile
# sed -i 's/CUDNN=0/CUDNN=1' Makefile
# sed -i 's/OPENCV=0/OPENCV=1' Makefile
# sed -i 's!/usr/local/cudnn/!/usr/local/cuda/!' Makefile
# cat Makefile
# make
# wget https://github.com/AlexeyAB/darknet/releases/download/
# darknet_yolo_v3_optimal/yolov4.weights
# ls -alrt
# wget https://anyimage of car or something

from IPython.display import Image
Image('image.jpg')

# ./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights
# -thresh 0.25 <image.jpg> --gpu

Image('predictions.jpg')

# wget https://image2.jpg

Image('image2.jpg')

# ./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights
# -ext_output <image2.jpg> --gpu -dont_show

#########################################################################################