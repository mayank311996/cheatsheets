from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
import streamlit as st
from PIL import Image
from utils_detection import draw_image_with_boxes

##############################################################################
st.set_option('deprecation.showfileUploaderEncoding', False)
st.header("Upload here for Face Detection")

uploaded_file = st.file_uploader("Choose an image...",
                                 key="1")  # , type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image.save('test.jpg')
    st.image(image, caption='Uploaded Image for Face Detection',
             width=300)
    st.write("")
#    st.write("Predicted:")

    # load image from file
    pixels = pyplot.imread('test.jpg')
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    faces = detector.detect_faces(pixels)
    print(faces)
    # display faces on the original image
    draw_image_with_boxes('test.jpg', faces)
    st.write(f"Number of faces: {len(faces)}")
    st.write("")
    st.write("")
    result = Image.open('result.png')
    st.image(result, caption='Detected Faces',
             width=300)

##############################################################################