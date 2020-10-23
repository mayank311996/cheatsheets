## Run 

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
```

Edit all the files. (Check alrady edited `serverless.yml` and `handler.py`)

Deploy

```bash
sudo sls deploy 
```

After this configure the API gateway (follow [medium blog](https://towardsdatascience.com/aws-lambda-amazon-api-gateway-not-as-daunting-as-they-sound-part-1-d77b92f53626) for this)

## Resources 

