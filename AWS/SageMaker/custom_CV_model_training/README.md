## Run

Access the article [here](https://www.sicara.ai/blog/amazon-sagemaker-model-training)

- Tool needed: [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html), Docker, [Kaggle CLI](https://github.com/Kaggle/kaggle-api)

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




















