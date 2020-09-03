## Run

Access the article [here](https://www.sicara.ai/blog/amazon-sagemaker-model-training)

- Tool needed: [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html), [Docker](https://docs.docker.com/engine/install/ubuntu/), [Kaggle CLI](https://github.com/Kaggle/kaggle-api)

- Some resources:
    1. https://github.com/awslabs/amazon-sagemaker-examples
    2. https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html
    3. https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html
    4. https://github.com/aws/deep-learning-containers

> STEP 1
#### Setup AWS credentials 

```bash
cat .aws/credentials
aws configure
aws configure list-profiles
```

- We set default region to `us-east-2`. 
- In this case we are using default profile. Other profiles can be
used by specifying `export AWS_PROFILE=<profile name>`.
- Also we are using credentials of `ml_serverless` profile as it also 
has all the actions. This can be changed later for security purpose as 
by default it has Administrator access. 

> STEP 2
#### Creating S3 buckets and loading data 

To create buckets (by default buckets are created in us-east-2; 
remember data and SageMaker should be in the same region otherwise it 
won't work)
```bash
aws s3 mb s3://template-sagemaker-cv-custom-data
aws s3 mb s3://template-sagemaker-cv-custom-model
```

Downloading and uploading data into s3 bucket
```bash
kaggle datasets download -d alxmamaev/flowers-recognition
unzip flowers-recognition.zip
aws s3 sync flowers s3://template-sagemaker-cv-custom-data
```

> STEP 3
#### Writing train.py and requirements.txt

```bash
cat train.py
cat requirements.txt
```

> STEP 4
#### Creating docker image for training container

Creating dockerfile 
```bash
cat Dockerfile
sudo systemctl status docker
```

Retrieving ECR authentication token 
```bash
aws ecr get-login-password | docker login --username AWS --password-stdin AWS_ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com
```
In my case
```bash
aws ecr get-login-password | docker login --username AWS --password-stdin 882207224999.dkr.ecr.us-east-2.amazonaws.com
```

To fix the permission denied [error](https://www.digitalocean.com/community/questions/how-to-fix-docker-got-permission-denied-while-trying-to-connect-to-the-docker-daemon-socket)
```bash
docker ps -a
sudo usermod -aG docker ${USER}
su -s ${USER}
docker ps -a
```

or (this worked in our case)
```bash
sudo chmod 666 /var/run/docker.sock
```

After login you should see this output (check the warning!)
```bash
WARNING! Your password will be stored unencrypted in /home/venom/.docker/config.json.
Configure a credential helper to remove this warning. See
https://docs.docker.com/engine/reference/commandline/login/#credentials-store

Login Succeeded
```

Creating ECR repo named `sagemaker-images`
```bash
aws ecr create-repository --repository-name sagemaker-images
```

Keep repositoryUri displayed in terminal
Create docker image and push it to the ECR repo
```bash
docker build -t REPOSITORY_URI . 
docker push REPOSITORY_URI
```
In my case
```bash
docker build -t 882207224999.dkr.ecr.us-east-2.amazonaws.com/sagemaker-images . 
docker push 882207224999.dkr.ecr.us-east-2.amazonaws.com/sagemaker-images
```

> STEP 5
#### Creating role for SageMaker training jobs








