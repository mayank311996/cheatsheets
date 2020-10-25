import boto3
s3 = boto3.resource('s3')
tar_file = 'model.tar.gz'
# replace 'mybucket' with the name of your S3 bucket
s3.meta.client.upload_file(
    tar_file,
    'fgc-trained-models-sagemaker',
    'car_make_model_fastai_resnet/model.tar.gz'
)
