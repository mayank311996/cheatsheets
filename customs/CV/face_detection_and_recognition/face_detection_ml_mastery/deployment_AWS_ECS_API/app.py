from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
import streamlit as st
from PIL import Image
from utils_detection import draw_image_with_boxes
from utils_detection import write_to_file
from utils_detection import allowed_file
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import os

##############################################################################
app = Flask(__name__)

UPLOAD_FOLDER = '/test'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


@app.route('/api_predict', methods=['POST', 'GET'])
def api_predict():
    if request.method == 'GET':
        return "Please Send POST Request"
    elif request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'test.jpg'))

        # load image from file
        pixels = pyplot.imread('/test/test.jpg')
        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        faces = detector.detect_faces(pixels)
        # print(faces)
        # display faces on the original image
        draw_image_with_boxes('/test/test.jpg', faces)

        return str(len(faces))


##############################################################################
if __name__ == "__main__":
    app.run(host='0.0.0.0')

##############################################################################
