import sagemaker
from sagemaker import get_execution_role
from sagemaker.predictor import csv_serializer, json_deserializer
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
    base_job_name='v1-xgboost-diabetes'
)

estimator.set_hyperparameters(
    max_depth=7,
    objective='binary:logistic',
    num_round=100  # same as n_estimators in sklearn
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
# deploy the model
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge',
    endpoint_name='v2-xgboost-diabetes'
)

predictor.serializer = csv_serializer
predictor.deserializer = None
# predictor.content_type = 'text/csv'

data = [4, 9.4, 0.5, 2, 0, 2, 1, 2]

result = predictor.predict(data)
np.round(float(result))

predictor.delete_endpoint()

#########################################################################################




















