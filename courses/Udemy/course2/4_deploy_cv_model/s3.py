import boto3, os, tempfile
from keras.applications.resnet50 import ResNet50

#########################################################################################

MODEL_BUCKET_NAME = 'ml-model-3521'
MODEL_KEY_NAME = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'

IMG_UPLOAD_BUCKET_NAME = 'image-uploads-3521'
IMG_KEY_NAME = 'elephant.jpg'

temp_dir = 'tmp'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

s3 = boto3.resource('s3')

# %% time
model_path = os.path.join(temp_dir, MODEL_KEY_NAME)
print('Downloading model...')
s3.Bucket(MODEL_BUCKET_NAME).download(MODEL_KEY_NAME, model_path)
print('Model downloaded')

# %% time
print('Loading model...')
model = ResNet50(weights=model_path)
print('Model loaded')

print('Downloading image...')
tmp_file = tempfile.NamedTemporaryFile(dir=temp_dir, delete=False)
img_object = s3.Bucket(IMG_UPLOAD_BUCKET_NAME).Object(IMG_KEY_NAME)
img_object.download_fileobj(tmp_file)
tmp_file.close()
print('Image downloaded to', tmp_file.name)

#########################################################################################


