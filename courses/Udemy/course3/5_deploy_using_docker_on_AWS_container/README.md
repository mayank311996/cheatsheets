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















