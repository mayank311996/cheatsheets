import base64
import io
from PIL import Image
import requests, json
import numpy as np
import cv2
import matplotlib.pyplot as plt

##############################################################################
# To convert image to string
with open("sample_images/dog_with_ball.jpg", "rb") as image_file:
    base64str = base64.b64encode(image_file.read()).decode("utf-8")


##############################################################################
# To convert string back to image
def base64str_to_PILImage(base64str):
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    bytesObj = io.BytesIO(base64bytes)
    img = Image.open(bytesObj)
    return img


##############################################################################
# To send request to deployed API
payload = json.dumps({
    "base64str": base64str,
    "threshold": 0.5
})

response = requests.put(
    "[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)",
    data=payload)
data_dict = response.json()
print(data_dict)


##############################################################################
# To visualize received response from deployed API
def PILImage_to_cv2(img):
    return np.asarray(img)


def drawboundingbox(img, boxes, pred_cls, rect_th=2, text_size=1, text_th=2):
    img = PILImage_to_cv2(img)
    class_color_dict = {}

    # initialize some random colors for each class for better looking
    # bounding boxes
    for cat in pred_cls:
        class_color_dict[cat] = [random.randint(0, 255) for _ in range(3)]

    for i in range(len(boxes)):
        cv2.rectangle(img, (int(boxes[i][0][0]), int(boxes[i][0][1])),
                      (int(boxes[i][1][0]), int(boxes[i][1][1])),
                      color=class_color_dict[pred_cls[i]], thickness=rect_th)
        cv2.putText(img, pred_cls[i],
                    (int(boxes[i][0][0]), int(boxes[i][0][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, text_size,
                    class_color_dict[pred_cls[i]],
                    thickness=text_th)  # Write the prediction class
    plt.figure(figsize=(20, 30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


# img = Image.open("sample_images/dog_with_ball.jpg")  # For local file
img = base64str_to_PILImage(base64str)  # For received image string from API
drawboundingbox(img, data_dict['boxes'], data_dict['classes'])


##############################################################################