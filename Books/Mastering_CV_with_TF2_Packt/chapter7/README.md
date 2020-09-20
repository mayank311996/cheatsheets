#### Detecting Objects using Darknet

Clone github repo
```bash
git clone https://github.com/pjreddie/darknet.git
```
Change to darknet directory and make file 
```bash
cd darknet 
make
```
Getting weights and making inference on sample image
```bash
wget https://pjreddie.com/media/files/yolov3.weights
./darknet detect cfg/yolov3.cfg yolov3.weights data/carhumanbike.png
```

#### Detecting Objects using Tiny Darknet 

Get weights 
```bash
cd darknet 
wget https://pjreddie.com/media/files/tiny.weights 
```
Inference 
```bash
./darknet detect cfg/tiny.cfg tiny.weights data/carhumanbike.png
./darknet classify cfg/tiny.cfg tiny.weights data/dog.png
```

#### Real-time prediction using Darknet (on video)

```bash
cd darknet 
sudo apt-get install libopencv-dev 
```
- Open `makefile` and change `OpenCV = 1`

Get the weights and recompile 
```bash
wget https://pjreddie.com/media/files/yolov3.weights 
make
```
Download and predict 
```bash
./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights data/road_video.mp4
```
- Open `makefile` and change `GPU` to 1
- Save makefile, remake and predict
- Open `yolov3.cfg` and change width and height if memory error is 
encountered
- Save makefile, remake and predict 

#### Custom Model Training 

- Follow section `preparing images` on page 209
- Follow section `generating annotation files` on page 210
- Follow section `converting .xml files to .txt files` on page 211
- Follow section `creating a combined train.txt and test.txt file` on page 212
- Follow section `creating a list of class name files` on page 212
- Follow section `creating a YOLO.data file` on page 212
- Follow section `adjusting the YOLO configuration file` on page 213
- Follow section `enabling the GPU for training` on page 215
- Follow section `start training` on page 216

 















