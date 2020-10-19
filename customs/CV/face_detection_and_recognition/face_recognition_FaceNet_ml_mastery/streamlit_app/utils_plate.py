import os
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN

# remove warning message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


##############################################################################
def extract_face(filename, required_size=(160, 160)):
    """
    This function extracts face from a given photograph
    :param filename: Input image filename
    :param required_size: Required size of output image
    :return: Cropped face from photograph
    """
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


def get_embedding(model, face_pixels):
    """
    This function converts cropped face into face embeddings
    :param model: Embedding model
    :param face_pixels: Cropped face
    :return: Face embeddings
    """
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]
##############################################################################
