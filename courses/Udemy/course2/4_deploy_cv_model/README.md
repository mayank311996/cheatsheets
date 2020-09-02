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

- Read about iamRoleStatements option in serverless.yml file in detail. 
For example s3:* is used for any operation on s3 bucket (Resource: *) 
too. 
- slimPatterns in custom section in serverless.yml allows you to specify
additional directories to remove from install packages. For example we 
don't need tensorboard during prediction so that can be mentioned here.
This can be also written in exclude section under package. 

```bash
sls invoke local --function resnet50-classify --path event.json
```

This time we will add requirements.txt in different way as the 
tensorflow package is big. The thing is in previous environment we 
installed jupyter and other packages which are not needed now.

```bash
conda create -n keras-deploy python=3.6
conda activate keras-deploy
pip install tensorflow==1.12.0 keras=2.2.4
pip freeze
pip freeze >> requirements.txt
```
 