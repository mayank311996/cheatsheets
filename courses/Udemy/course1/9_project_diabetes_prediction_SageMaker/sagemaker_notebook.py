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
train_data = r'diabetes/training/'
val_data = r'diabetes/validation/'

s3_model_output_location = r's3://{0}/diabetes/model'.format(bucket_name)
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

data = './diabetes.csv'
df = pd.read_csv(data)
print(df.shape)

cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
print(cols)

df = df[cols]
print(df.head())

train = df[:629]
val = df[629:]

print(train.shape)
print(val.chape)
print(train.head())
train.isnull().sum()

np_train = train.values
np_val = val.values
type(np_train)

np.savetxt('train.csv', np_train, delimiter=',')
np.savetxt('val.csv', np_val, delimiter=',')


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





















