import base64
import io
from PIL import Image

##############################################################################
# To convert image to string
with open("sample_images/dog_with_ball.jpg", "rb") as image_file:
    base64str = base64.b64encode(image_file.read()).decode("utf-8")


# To convert string back to image
def base64str_to_PILImage(base64str):
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    bytesObj = io.BytesIO(base64bytes)
    img = Image.open(bytesObj)
    return img
