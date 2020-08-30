## Run

original github [repo](https://github.com/mpavlovic/serverless-machine-learning)

> STEP 1
#### Installation - Miniconda

- [Miniconda install](https://docs.conda.io/en/latest/miniconda.html#linux-installers)

After downloading appropriate *.sh file
```
bash Miniconda*.sh  
```

> STEP 2
#### Installation - Docker

- Download three installation files from this url: [Docker](https://download.docker.com/linux/ubuntu/dists/bionic/pool/stable/amd64)
    - containerd.io_1.2.6-3_amd64.deb
    - docker-ce-cli_19.03.9~3-0~ubuntu-bionic_amd64.deb
    - docker-ce_19.03.9~3-0~ubuntu-bionic_amd64.deb  

```
sudo dpkg -i containerd.io_1.2.6-3_amd64.deb
sudo dpkg -i docker-ce-cli_19.03.9~3-0~ubuntu-bionic_amd64.deb
sudo dpkg -i docker-ce_19.03.9~3-0~ubuntu-bionic_amd64.deb
sudo systemctl enable docker
sudo docker run hello-world
```

> STEP 3
#### Installation - Serverless

This is second method. Prefer one from course1. 
```
sudo apt install curl
curl -sL https://deb.nodesource.com/setup_10.x -o nodesource_setup.sh
sudo bash nodesource_setup.sh
sudo apt-get install -y nodejs
nodejs -v
npm -v
sudo npm install -g severless
serverless
```

After this configure the serverless as mentioned in course1.
```
cd .aws
cat credentials
```