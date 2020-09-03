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
sudo sls package
```

- Now go through section 8, video 59.
- Show hidden files. Extract .requirements.zip and copy tensorflow folder 
outside. Delete .requirements.zip and extracted folder. 
- In most cases this is not necessary but in this case our zip requirement
will have more than 300MB after unzipping to the tmp directory and 
tensorflow alone would have around 160MB except requirements we will
also download a large model about 100MB and together with unzip requirements 
this will take more than 400MB in tmp directory, whose space is limited to 
500MB in theory but in practice it won't work (will give memory error).
- Luckily tensorflow package without it's dependencies is less than 
250MB so it can be moved to the root of deployment package folder.
- Will do the same for pillow 
- Go to `miniconda3/envs/keras-dev/lib/python3.6/site-packages`, 
copy pillow to the root folder of serverless project where we 
just pasted the tensorflow folder. 
- For pillow if we put in requirements.txt, you will get error in 
Lambda function saying cannot import PIL (pillow), so to avoide that
we explicitly copied and pasted it. (Not for size as in case of 
tensorflow)

- Now delete tensorflow==1.12.0 from requirements.txt as we already 
have it in the root folder. 

```bash
sudo sls deploy
sls invoke --function resnet50-classify --path event.json --log  
```

- Now setup the web-gui-code. Go through section 8, video 61. Also, setup
the AWS Cognito service as shown in the video.




 