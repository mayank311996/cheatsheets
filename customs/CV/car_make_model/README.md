## Run

To copy files from pc to AWS ec2
```bash
scp -i venom_key_pair_useast2.pem /home/venom/Downloads/dataset.py ubuntu@ec2-3-15-152-93.us-east-2.compute.amazonaws.com:~/car_make_model_train/
```

To unzip 
```bash
unzip file.zip
```

To activate pytorch env
```bash
source activate pytorch_p36
```

Running Jupyter notebook from AMI
```bash
ssh -L localhost:8888:localhost:8888 -i <your .pem file name> ubuntu@<Your instance DNS>
```

## Requirements
- Python3
- numpy
- pytorch
- torchvision
- scikit-learn
- matplotlib
- pillow
- torch (pytorch)
- torchvision

You can install the requirements using:
```
pip3 install -r requirements.txt
```

**Troubleshooting**: if you get some errors about pytorch or torchvision install use `sudo` to install it.

## Usage

First, if you have no resnet152 model trained and you need from scratch to do it you need to:

- download dataset
- preprocess the dataset
- train the model

After you can try a new sample.

### Download dataset

I suggest to use [VMMRdb](http://vmmrdb.cecsresearch.org/) as dataset, it's free and full of labelled images for car model recognition instead of detection (the most dataset is for this).

So download the dataset, select some models and put the directory model in the dataset folder, any directory in "dataset" will be considered a new class.

If you need more data for your project you can also add the followings dataset:
- [Stanford Cars Dataset from jkrause](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) (low images quantity)
- [Comprehensive Cars Database](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/), here the module to get this dataset [MODULE](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/agreement.pdf)

### Handle CSV training, testing, validation and dataset structure

The dataset structure should be like this:
```
dataset / classes / file.jpg
```

For example, we have 3 classes: **honda_civic, nissan and ford**:
```
dataset_dir / honda_civic / file1.jpg
dataset_dir / honda_civic / file2.jpg
....
dataset_dir / nissan / file1.jpg
dataset_dir / nissan / file2.jpg
....
dataset_dir / ford / file1.jpg
dataset_dir / ford / file2.jpg
...
and so on.
```

The **"dataset_dir"** is the **IMAGES_PATH** in config.py.
The python script will save the classes in a  dict() named **num_classes**, like this:
```
num_classes = {
  "honda_civic": 1,
  "nissan": 2,
  "ford": 3
}
```

This conversion happens automatically when you just add a directory inside the IMAGES_PATH, if you add tomorrow a new car, like, FIAT, the program will add automatically to the classes, just **pay attention to the order of the classes inside num_classes and the related trainin,testing and validation CSV files**.

The file **training, testing and validation (CSV)** should contain only two columns:
**FILE_NAME, NUM_CLASS**

Example of CSV file:
```
file1.jpg, 1
file2.jpg, 1
file1.jpg, 2
file2.jpg, 2
file1.jpg, 3
file2.jpg, 3
```

Anyway, this paragraph is only for your info, the CSV files are automatically genrated by the preprocessing phase explained in the follow paragraph.


### Preprocess the dataset
You have to generate the CSV files and calculate the mean and standard deviation to apply a normalization, just use the -p parameter to process your dataset so type:

```
$ python3 main.py -p
```

### Train the model

**Little introduction**

Before the training process, modify the `EPOCHS` parameter in `config.py`, usually with 3 classes 30-50 epochs should be enough, but you have to see the results_graph.pn file (when you finish your training with the default epochs parameter) and check if the blue curve is stable.

An example of the graph could be the follow:
![graph results - Car Model Recognition](https://user-images.githubusercontent.com/519778/67412403-81fe5c00-f5bf-11e9-9bd1-e86251bb9a0c.png)

After 45-50 epochs (number bottom of the graph), the blue curve is stable and does not have peaks down.
Moreover, the testing curve (the orange one) is pretty "stable", even with some peaks, for the testing is normal that the peaks are frequently.

**Train the model**

To train a new model resnet152 model you can run the main.py with the -t parameter, so type:

```
$ python3 main.py -t
```

The results will be saved in the results/ directory with the F1 score, accuracy, confusion matrix and the accuracy/loss graph difference between training and testing.

## Try new sample

To try predict a new sample you can just type:
```
python3 main.py -i path/file.jpg
```

## Resources 

- [original repo](https://github.com/Helias/Car-Model-Recognition/blob/master/README.md)

## Additional resources 

- https://github.com/spectrico/car-make-model-classifier-yolo3-python
- https://github.com/faezetta/VMMRdb
- https://www.dropbox.com/s/uwa7c5uz7cac7cw/VMMRdb.zip?dl=0
- https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-pytorch.html
