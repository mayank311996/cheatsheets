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
# Start the Training
sess = sagemaker.Session()
role = get_execution_role()
print(role)

container = sagemaker.amazon.amazon_estimator.get_image_uri(
    sess.boto_region_name,
    "xgboost",
    "latest"
)
print(f"SageMaker XGBoost Info : {container} ({sess.boto_region_name})")

# Building the model
estimator = sagemaker.estimator.Estimator(
    container,
    role,
    train_instance_count=1,
    train_instance_type='ml.m4.xlarge',
    output_path=s3_model_output_location,
    sagemaker_session=sess,
    base_job_name='v1-xgboost-bcancer'
)

estimator.set_hyperparameters(
    max_depth=3,
    objective='binary:logistic',
    num_round=500  # same as n_estimators in sklearn
)
print(estimator.hyperparameters())

# specify the files for training and validation
training_input_config = sagemaker.session.s3_input(
    s3_data=s3_training_file_location,
    content_type='csv',
    s3_data_type='S3Prefix'
)
validation_input_config = sagemaker.session.s3_input(
    s3_data=s3_validation_file_location,
    content_type='csv',
    s3_data_type='S3Prefix'
)

data_channels = {
    'train': training_input_config,
    'validation': validation_input_config
}
print(training_input_config.config)
print(validation_input_config.config)

# start the training
estimator.fit(data_channels)

#########################################################################################









