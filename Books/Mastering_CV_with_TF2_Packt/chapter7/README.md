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

