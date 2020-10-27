FROM ubuntu:latest
RUN apt-get update
RUN apt-get install python3.6
RUN apt install python3-pip
RUN python3 -m pip install --upgrade pip
RUN pip3 install scikit-learn
RUN pip3 install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.3.0-cp36-cp36m-manylinux2010_x86_64.whl
RUN pip3 install mtcnn
RUN pip3 install matplotlib
RUN pip3 install boto3
RUN mkdir test
COPY ./app.py app.py
COPY ./utils_detection.py utils_detection.py
EXPOSE 5000
ENTRYPOINT [ "python3" ]
CMD [ "app.py" ]

