## Run
> STEP 1
#### Building docker container from dockerfile 

Docker will search dockerfile in the present folder itself if we put "." at end
```
sudo docker build -t mpi_api_final .
```

> STEP 2
#### Display docker images 

```
docker images
```

> STEP 3
#### TO remove running docker container and the image

```
sudo docker rm `container_num`
sudo docker rmi `image_name`
```
