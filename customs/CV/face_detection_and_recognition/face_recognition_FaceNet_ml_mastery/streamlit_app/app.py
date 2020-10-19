import os
import streamlit as st
import joblib
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.preprocessing import Normalizer
from utils_face import extract_face
from utils_face import get_embedding

# remove warning message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

##############################################################################
# Load trained sklearn pipeline
out_encoder = joblib.load('out_encoder.pkl')
# Load Classifier sklearn model
model_classifier = joblib.load("model_classifier.pkl")
# Load embeddings model
model = load_model('facenet_keras.h5')

st.set_option('deprecation.showfileUploaderEncoding', False)
st.header("Upload here for Face Recognition")

uploaded_file = st.file_uploader("Choose an image...",
                                 key="1")  # , type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image.save('test.jpg')
    st.image(image, caption='Uploaded Image for Face Recognition',
             width=300)
    st.write("")
    st.write("Predicted:")

    face = extract_face('test.jpg')
    embeddings = get_embedding(model, face)

    in_encoder = Normalizer(norm='l2')
    embeddings = embeddings.reshape(1, -1)
    X = in_encoder.transform(embeddings)

    y_class = model_classifier.predict(X)
    y_proba = model_classifier.predict_proba(X)

    class_index = y_class[0]
    class_probability = y_proba[0, class_index] * 100
    predict_names = out_encoder.inverse_transform(y_class)

    st.write(f"Predicted: {predict_names}, Probability: {class_probability}")
    st.image(face, caption='Cropped Face',
             width=300)

##############################################################################
