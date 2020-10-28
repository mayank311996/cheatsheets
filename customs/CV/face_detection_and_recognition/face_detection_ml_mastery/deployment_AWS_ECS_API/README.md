## Run 

```bash
sudo docker build -t face_detection_ml_mastery .
sudo docker images
sudo docker tag face_detection_ml_mastery 882207224999.dkr.ecr.us-east-2.amazonaws.com/fgc-phase1
sudo docker images
sudo docker push 882207224999.dkr.ecr.us-east-2.amazonaws.com/fgc-phase1
```

After this follow steps mentioned [here](https://github.com/mayank311996/cheatsheets/tree/master/courses/Udemy/course3/5_deploy_using_docker_on_AWS_container#run-docker-image-on-amazon-container-service-ecs)

## To Do 

- later incorporate requirements.txt rather than typing manually in 
dockerfile 
- edit app.py. Remove all comments and add functions 

## Resources 

- https://jdhao.github.io/2020/04/12/build_webapi_with_flask_s2/
- https://jdhao.github.io/2020/03/17/base64_opencv_pil_image_conversion/
- https://stackoverflow.com/questions/26149318/pillow-oserror-cannot-identify-image-file-io-bytesio-object-at-0x02345ed8
- https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/