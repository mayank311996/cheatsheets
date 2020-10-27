## Run

We will use approach mentioned [here](https://github.com/mayank311996/cheatsheets/tree/master/customs/CV/car_make_model/car_make_model_fastai_resnet/deployment_AWS_Lambda_API_success)

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
    - Path: /ml
    - In Root directory permission section
        - Owner ID: UID from Lambda eg. 994
        - Owner group ID: 994
        - Permissions: 777
- Check network section for created EFS, if empty fill details 

> STEP 3 
> Creating EC2 instance 

- This is needed to fill EFS with dependencies and trained ML model
- Name: Lambda_EFS_FUNCTIONNAME
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
pip3 install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.3.0-cp36-cp36m-manylinux2010_x86_64.whl
pip3 install mtcnn
pip3 install matplotlib
pip3 install boto3
pip3 install requests
sudo cp -r /home/ubuntu/.local/lib/python3.6/site-packages/* /mnt/efs/fs1/ml/lib/
sudo cp -r /usr/lib/python3/dist-packages/* /mnt/efs/fs1/ml/lib/
```

exit ec2

> STEP 4
> Configuring Lambda for EFS 

- Go to function and under it to file system tab
    - Attach file system
    - Set local mount path to `/mnt/inference`
- Configure VPC section
    - select all subnets and security group as similar to EFS
- In environment variable add 
    - Key: PYTHONPATH, Value:/mnt/inference/lib

> STEP 5 
> Checking and deploying code into Lambda

- First try this

```
...
import torch 
...

...
return {
    'statusCode': 200,
    'body': json.dump(f"Pytorch version: {torch.__version__}")
}
...
```         

If above code runs without error that means we mounted EFS successfully

Now copy and paste code from `handler.py` into Lambda function

Click on deploy

> STEP 6
> Setting up API Gateway

- Follow this [article](https://towardsdatascience.com/aws-lambda-amazon-api-gateway-not-as-daunting-as-they-sound-part-1-d77b92f53626) to setup API gateway 

> STEP 7
> Test using Postman

- Follow this [article](https://towardsdatascience.com/aws-lambda-amazon-api-gateway-not-as-daunting-as-they-sound-part-1-d77b92f53626)

Enjoy!

## To Do

- To improve inference time extract models in EFS rather that extracting 
every time in Lambda function code. 

## Resources 

- https://medium.com/@mikepalei/serving-a-tensorflow-2-model-on-aws-lambda-58ce64ef7d38
- https://www.tensorflow.org/install/pip#package-location
- https://towardsdatascience.com/aws-lambda-amazon-api-gateway-not-as-daunting-as-they-sound-part-1-d77b92f53626
- https://medium.com/@rajputankit22/upgrade-python-2-7-to-3-6-and-3-7-in-ubuntu-97d2727bf911
- https://help.dreamhost.com/hc/en-us/articles/115000699011-Using-pip3-to-install-Python3-modules
- https://aws.amazon.com/blogs/compute/building-deep-learning-inference-with-aws-lambda-and-amazon-efs/
