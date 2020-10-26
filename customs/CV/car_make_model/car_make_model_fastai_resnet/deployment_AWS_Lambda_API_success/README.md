## Run 

In this deployment we are following [this](https://medium.com/@mikepalei/serving-a-tensorflow-2-model-on-aws-lambda-58ce64ef7d38) article. 

By default we are choosing default VPC for our deployment and choosing 
default subnets and all zones in a region. Later when in production this 
can be changed according to [this](https://medium.com/@mikepalei/serving-a-tensorflow-2-model-on-aws-lambda-58ce64ef7d38) article for security purpose. 

> STEP 1 
> Creating a Lambda function 

- Create a Lambda function with following details
    - Runtime: python3.6
    - Memory: Max (3GB)
    - Time out: 30 secs
    - Have default role (First time) or existing role (Not first time) 

- Create function 

- Go to IAM roles (Only for first time) and add these policies to AWS
Lambda role
    - S3 Full access 
    - AWS Lambda VPC Access Execution Role 
    - Amazon Elastic File System Client Read Write Access 
    
- Get User id by running sample snippet in Lambda function. We will need 
that to configure the EFS.

```
return {
    'statusCode': 200,
    'body': json.dump(f"Current user id: {os.getuid()}")
}
```

> STEP 2 
> Creating a EFS 

- Create a EFS with name as extended version of Lambda function name
    - Name: FUNCTIONNAME_Lambda_EFS
    - Choose same VPC as Lambda
- Create access point
    - Choose created file system
    - Name: Lambda_access_point_FUNCTIONNAME
    - In Root directory permission section
        - Owner ID: UID from Lambda eg. 994
        - Owner group ID: 994
        - Permissions: 777
- Check network section for created EFS, if empty fill details 

> STEP 3 
> Creating EC2 instance 

- This is needed to fill EFS with dependencies and trained ML model
- Create EC2 instance with t2.xlarge and attach EFS in step 3
    - Select same security group as created EFS 
    
- Now SSH into EC2 instance 

After logging in
```bash
mount
```
and check if `/mnt/efs/fs1` is present among listed path (cross verification of mounted EFS)

Write following commands 
```bash
sudo mkdir /mnt/efs/fs1/ml
sudo chown ubuntu:ubuntu /mnt/efs/fs1/ml
sudo apt-get update 
sudo apt-get install python3.6
sudo apt install python3-pip
pip3 --version
python3 -m pip install --upgrade pip
pip3 install scikit-learn
pip3 show scikit-learn
pip3 install torch
pip3 install torchvision
pip3 install boto3
pip3 install requests
sudo cp -r /home/ubuntu/.local/lib/python3.6/site-packages/* /mnt/efs/fs1/ml/lib/
sudo cp -r /usr/lib/python3/dist-packages/* /mnt/efs/fs1/ml/lib/
```

Create a `S3 bucket`. Need to do only first time
(In our case the name is "fgc-trained-models-sagemaker")

```bash
aws s3 mb s3://REPLACE_WITH_YOUR_BUCKET_NAME
```

For reference, check `Exporting model and other dependencies for 
deployment on AWS Lambda` section in `car_make_model_fastai.ipynb`
under `deployment_AWS_Lambda_API/notebooks` directory

Download the model.tar.gz file from ec2 instance 

Run `upload_s3.py` script 

Download and explore example project from fastai

```bash
wget https://github.com/fastai/course-v3/raw/master/docs/production/aws-lambda.zip
unzip aws-lambda.zip
```

Create serverless project 

```bash
sls create --template aws-python3 --name car_make_model
sls plugin install -n serverless-python-requirements@4.2.4
```

Edit all the files. (Check already edited `serverless.yml` and `handler.py`)

We will follow approach highlighted [here](https://github.com/mayank311996/cheatsheets/tree/master/courses/Udemy/course2/4_deploy_cv_model) to deploy dependencies as they
are too large

Create a conda environment (only once)
```bash
conda create -n pytrochdeploy python=3.6
conda activate pytrochdeploy 
```

Now install only CPU version of pytroch (GPU version is 1.2GB, while CPU version is around 300MB)
(As Lambda max size is 500MB, GPU version won't fit)

```bash
pip install torch==1.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

Now copy `torch`, `PIL` and `torchvision` from `/home/anaconda3/envs/pytrochdeploy/
lib/python3.6/site-packages/torch/lib`

Create a package with 

```bash
sls package
sls deploy
```

It will still not work as deployment package will be larger than 500MB.
Searching for some other way. 

Deploy

```bash
sudo sls deploy 
```

After this configure the API gateway (follow [medium blog](https://towardsdatascience.com/aws-lambda-amazon-api-gateway-not-as-daunting-as-they-sound-part-1-d77b92f53626) for this)

## Note

- Previously we were planning to use public ARN in Lambda function. Layer: `arn:aws:lambda:us-east-2:934676248949:layer:pytorchv1-py36:2`
- However, that is pretty old and uses `pytorch 1.1.0` while we need `pytroch 1.6.0`
otherwise we will get some "module initialization error (Cloud Log)"
- So, now implementing as `requirements.txt` way 
- This will not work as well because size of dependencies will be too large
- So we will follow this [way](https://github.com/mayank311996/cheatsheets/tree/master/courses/Udemy/course2/4_deploy_cv_model)
- Even this is still large and more than 500MB. Need to check for some other way.

## To Do

- To improve inference time extract models in EFS rather that extracting 
every time in Lambda function code. 

## Resources 

- https://medium.com/@mikepalei/serving-a-tensorflow-2-model-on-aws-lambda-58ce64ef7d38
- https://medium.com/@rajputankit22/upgrade-python-2-7-to-3-6-and-3-7-in-ubuntu-97d2727bf911
- https://help.dreamhost.com/hc/en-us/articles/115000699011-Using-pip3-to-install-Python3-modules

