import sagemaker
from sagemaker import get_execution_role
import numpy as np
import pandas as pd
import boto3
import re
print("Libraries Loaded")

#########################################################################################
# Uploaded data to S3
bucket_name = 'awsdataforml'
train_data = r'bcancer/training/'
val_data = r'bcancer/validation/'

s3_model_output_location = r's3://{0}/bcancer/model'.format(bucket_name)
s3_training_file_location = r's3://{0}/{1}'.format(
    bucket_name,
    train_data
)
s3_validation_file_location = r's3://{0}/{1}'.format(
    bucket_name,
    val_data
)

print(s3_model_output_location)
print(s3_training_file_location)
print(s3_validation_file_location)


def write_to_s3(filename, bucket, key):
    with open(filename, 'rb') as f:
        return boto3.Session().resource('s3').Bucket(bucket)\
            .Object(key).upload_fileobj(f)


write_to_s3(
    'train.csv',
    bucket_name,
    train_data + 'train.csv'
)

write_to_s3(
    'test.csv',
    bucket_name,
    val_data + 'test.csv'
)

#########################################################################################













