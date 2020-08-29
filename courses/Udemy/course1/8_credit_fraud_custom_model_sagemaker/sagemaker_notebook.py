import os
import sys
import sagemaker
from sagemaker import get_execution_role
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sagemaker.sklearn.estimator import SKLearn

#########################################################################################
sagemaker_session = sagemaker.Session()
role = get_execution_role()
region = sagemaker_session.boto_session.region_name

train_file = 'trainCreditUS.csv'
test_file = 'testCreditUS.csv'

bucket_name = 'awsdataforml'
training_folder = r'fraudDetection/train'
test_folder = r'fraudDetection/test'
model_folder = r'fraudDetection/model/'

training_data_uri = r's3://' + bucket_name + r'/' + training_folder
testing_data_uri = r's3://' + bucket_name + r'/' + test_folder
model_data_uri = r's3://' + bucket_name + r'/' + model_folder

print(training_data_uri, testing_data_uri, model_data_uri)

sagemaker_session.upload_data(
    train_file,
    bucket=bucket_name,
    key_prefix=training_folder
)
sagemaker_session.upload_data(
    test_file,
    bucket=bucket_name,
    key_prefix=test_folder
)

# instance_type = 'ml.t2.medium'
instance_type = 'local'

estimator = SKLearn(
    entry_point='sklearn_fraud_detection.py',
    train_instance_type=instance_type,
    role=role,
    output_path=model_data_uri,
    base_job_name='fraud_detection_2',
    hyperparameters={
        'n_estimators': 100,
        'max_depth': 3
    }
)
estimator.fit(
    {
        'training': training_data_uri,
        'testing': testing_data_uri
    }
)

print(estimator.latest_training_job.job_name)
print(estimator.model_data)

predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type=instance_type
)

#########################################################################################















