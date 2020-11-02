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

```bash
|--dockerfastapi
   |-- Dockerfile
   |-- app
   |   |-- main.py
   |-- requirements.txt
```   

To build docker file 
```bash
sudo docker build -t myimage .
```

To start docker container using an image 
```bash
sudo docker run -d --name mycontainer -p 80:80 myimage
```

Type this in browser
```bash
<IPV4 public IP>/docs
```

## Troubleshooting 

To see logs 
```bash
sudo docker logs -f mycontainer
```

To start and stop docker 
```bash
sudo service docker stop
sudo service docker start
```

Listing images and containers
```bash
sudo docker container ls
sudo docker image ls
```

Deleting unused docker images and containers
the prune command removes the unused containers and images
```bash
sudo docker system prune
```

delete a particular container
```bash
sudo docker rm mycontainer
```

remove myimage
```bash
sudo docker image rm myimage
```

remove all images
```bash
sudo docker image prune — all
```

To check browser output in terminal
```bash
curl localhost/docs
```

Develop without reloading image again and again 
```bash
sudo docker run -d -p 80:80 -v $(pwd):/app myimage /start-reload.sh
```

## Streamlit app

```bash
streamlit run streamlitapp.py
```

## Note

- Sending request through postman won't work as it sends binary file 
not string version of the image.
 

## Resources 

- https://mlwhiz.com/blog/2020/08/08/deployment_fastapi/
- https://towardsdatascience.com/a-layman-guide-for-data-scientists-to-create-apis-in-minutes-31e6f451cd2f
- https://fastapi.tiangolo.com/deployment/
- https://pytorch.org/docs/stable/torchvision/models.html
- https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker/blob/master/docker-images/python3.7.dockerfile
- https://github.com/tiangolo/uvicorn-gunicorn-docker/blob/master/docker-images/python3.7.dockerfile
- https://towardsdatascience.com/how-to-write-web-apps-using-simple-python-for-data-scientists-a227a1a01582
- https://towardsdatascience.com/how-to-deploy-a-streamlit-app-using-an-amazon-free-ec2-instance-416a41f69dc3

