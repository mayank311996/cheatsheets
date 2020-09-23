## Run

To check all containers running currently and exited 
```bash
sudo docker ps -a 
```

To check all available images 
```bash
sudo docker images
```

Pulling an image (it first checks on local machine then goes to 
docker hub)
```bash
sudo docker pull hello-world
```

Running the image 
```bash
sudo docker run hello-world 
```

To remove docker container 
```bash
sudo docker rm <CONTAINER ID>
```

To remove docker image
```bash
sudo docker rmi <REPOSITORY>
```

### Setup Flower Deployment API on Docker Container

Running Ubuntu container 
```bash
sudo docker run -p 5000:5000 -it ubuntu bash
```

```bash
apt-get update
apt-get install -y python3-pip
pip3 install scikit-learn
pip3 install flask
```

In another terminal type
```bash
sudo docker cp flower-v1.pkl <CONTAINER ID>:flower-v1.pkl
sudo docker cp predict_flower.py <CONTAINER ID>:predict_flower.py
```

Back to container terminal
```bash
python3 predict_flower.py
```

Now if you send anything on port 5000, it will be redirected to 
container's port 5000 and will be executed on container 

This was just to check how docker works. Real stuff is below

### Building and pushing docker image

In new terminal type to build docker image from Dockerfile 
```bash
sudo docker build -t ml_flower_app .
```

Go to https://hub.docker.com/ and create account then type 
```bash
sudo docker login 
```

Tagging the image 
```bash
sudo docker tag ml_flower_app <USER NAME OF DOCKER HUB>/flower_app
```

To push image to docker hub
```bash
sudo docker push <USER NAME OF DOCKER HUB>/flower_app
```

### Run Docker image on Amazon Container Service (ECS)

Follow video 52 under section 11

- Go to ECS
- Click on `Get Started` 
- Choose `custom` inside `container definition` 
- Click on `configure` and set the fields 
- In `image` field type `<USER NAME OF DOCKER HUB>/flower_app`
- In `container name` type `flower_app`
- Set `memory limit` to `Soft limit` `512`
- Set `port mappings` to `5000`
- Click `next`
- Configure `Service` and `Cluster` sections
- Click on `create`
- After launch click on `view service`
- Go to `task` and get `public IP`

 








