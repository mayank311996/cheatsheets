## Run 

To train and get inference with YOLOv5 we follow [this](https://mlwhiz.com/blog/2020/08/08/yolov5/) article

We will use deep learning EC2 instance "Deep_learning_GPU_trial" for training 

SSH into EC2 

```bash
ssh -L localhost:8888:localhost:8888 -i "#############.pem" ubuntu@############.us-east-2.compute.amazonaws.com
```

```bash
mkdir custom_train
cd custom_train
mkdir YOLOv5_pytorch_ultralytics_mlwhiz
```

```bash
git clone https://github.com/EscVM/OIDv4_ToolKit
cd OIDv4_ToolKit
pip install -r requirements.txt
python3 main.py downloader --classes Cricket_ball  Football --type_csv all -y --limit 500
cd OID
ls
```

Create a notebook file called `data.ipynb` inside `OIDv4_Toolkit` folder

All files and folder are synced to this Github repo from EC2 instance for reference

Run `data.ipynb`

```bash
cd dataset 
ls -l . | egrep -c '^-'
```

Now we are good with the dataset. Everything is under one folder but later
we will create folders like train, validation and test according to YOLOv5
requirements and also delete previously downloaded duplicate data to save 
some space. 


## To Do

- Use Docker and SageMaker to train. First trained using EC2 because 
it's easy to debug in jupyter notebooks

## Resources 

- https://mlwhiz.com/blog/2020/08/08/yolov5/
- https://github.com/ultralytics/yolov5
- https://github.com/ultralytics/yolov3
- https://github.com/EscVM/OIDv4_ToolKit

