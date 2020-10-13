import cv2
import streamlit as st

from dataset import torch, os, LocalDataset, transforms, np, get_class, \
    num_classes, preprocessing, Image, m, s
from config import *

from torch import nn
from torch.optim import SGD
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.models import resnet, vgg

# remove warning message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

##############################################################################
st.header("Upload here for Car Make and Model Detection")

uploaded_file = st.file_uploader("Choose an image...",
                                 key="1")  # , type="jpg")

mean = m
std_dev = s

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std_dev)])

# Convert all this to a function later, otherwise model gets loaded all time
classes = {"num_classes": len(num_classes)}
resnet152_model = resnet.resnet152(pretrained=False, **classes)
model_name = "resnet152"
model = resnet152_model

# print(
#     str(RESULTS_PATH) + "/" + str(model_name) + "/" + str(model_name) + ".pt"
#     )
model.load_state_dict(torch.load(
    str(RESULTS_PATH) + "/" + str(model_name) + "/" + str(model_name) + ".pt"))

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image.save('test.jpg')
    st.image(image, caption='Uploaded Image for License Plate Detection',
             width=300)
    st.write("")
    st.write("Predicted:")

    # image = load_img(uploaded_file)
    # image = img_to_array(image)

    img = cv2.imread('test.jpg')
    # img = img_to_array(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = transform(img)

    cuda_available = torch.cuda.is_available()

    if USE_CUDA and cuda_available:
        model = model.cuda()
    model.eval()

    x = Variable(im.unsqueeze(0))

    if USE_CUDA and cuda_available:
        x = x.cuda()
        pred = model(x).data.cuda().cpu().numpy().copy()
    else:
        pred = model(x).data.numpy().copy()

    # print (pred)

    idx_max_pred = np.argmax(pred)
    idx_classes = idx_max_pred % classes["num_classes"]
    # print(get_class(idx_classes))

    st.write('%s' % get_class(idx_classes))

##############################################################################
