## Run

- Go through section 8, video 1. 

> STEP 1
#### Create conda environment 

```bash
conda create -n keras-dev python=3.6 pylint rope jupyter
conda activate keras-dev
pip install tensorflow==1.12.0 keras==2.2.4 boto3 pillow 
```

- create two s3 buckets. One for images and one for model. eg. 1) 
image-upload-3521, and 2) ml-models-3512

- To see how to programmatically access those objects saved in s3
buckets check s3.py

> STEP 2
>#### Create serverless project

```bash
sls create --template aws-python3 --name resnet50
sls plugin install -n serverless-python-requirements@4.2.4
```