FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install scikit-learn
RUN pip3 install flask
COPY ./flower-v1.pkl flower-v1.pkl
COPY ./predict_flower.py predict_flower.py
EXPOSE 5000
ENTRYPOINT [ "python3" ] 
CMD [ "predict_flower.py" ] 