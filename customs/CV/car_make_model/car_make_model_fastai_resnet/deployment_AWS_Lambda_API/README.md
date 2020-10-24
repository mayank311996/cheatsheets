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

## Resources 


