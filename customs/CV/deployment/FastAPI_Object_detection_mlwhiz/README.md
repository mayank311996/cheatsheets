## Tasks 

- Setting Up an Amazon Instance
- Creating a FastAPI API for Object Detection
- Deploying FastAPI using Docker
- An End to End App with UI

## Run

We will be using a Pytorch pre-trained fasterrcnn_resnet50_fpn detection model from the torchvision.models for object detection, which is trained on the COCO dataset to keep the code simple

If not installed already
```bash
pip install fastapi
pip install uvicorn
```

To run app
```bash
uvicorn fastapi_app:app --reload
```
Remove --reload flag when you put API in production 

Now it seems like app is working fine on local PC so it's time to move 
app on the EC2 instance 

Docker directory structure 
|--dockerfastapi
   |-- Dockerfile
   |-- app
   |   |-- main.py
   |-- requirements.txt

## Note

- Sending request through postman won't work as it send binary file 
not string version of the image.
 

## Resources 

- https://mlwhiz.com/blog/2020/08/08/deployment_fastapi/
- https://towardsdatascience.com/a-layman-guide-for-data-scientists-to-create-apis-in-minutes-31e6f451cd2f
- https://fastapi.tiangolo.com/deployment/
- https://pytorch.org/docs/stable/torchvision/models.html

