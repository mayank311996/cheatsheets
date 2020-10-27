from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from PIL import Image
from utils_detection import draw_image_with_boxes
from utils_detection import write_to_file
from utils_detection import allowed_file
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import io
import base64

##############################################################################
app = Flask(__name__)

# UPLOAD_FOLDER = '/test'
# ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


@app.route('/api_predict', methods=['POST', 'GET'])
def api_predict():
    if request.method == 'GET':
        return "Please Send POST Request"
    elif request.method == 'POST':
        im_file = io.BytesIO(request.get_data())
        img = Image.open(im_file)
        img.save('test.jpg')

        # file = request.files['image']
        # file.save('im-received.jpg')
        ## img = Image.open(file.stream)

        ## print(request.files['file'].read())
        ## write_to_file('test.jpg', request.files['file'])

        # load image from file
        pixels = pyplot.imread('test.jpg')
        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        faces = detector.detect_faces(pixels)
        # print(faces)
        # display faces on the original image
        # draw_image_with_boxes('/test/test.jpg', faces)

        return str(len(faces))


##############################################################################
if __name__ == "__main__":
    app.run(host='0.0.0.0')

##############################################################################
