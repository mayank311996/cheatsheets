## Steps 

1. Reading search term
2. Get all videos for given search term on Youtube first page.
3. Scrap views of every single video and return total views and 
average views. 

## Run

```bash
sls create --template aws-python3
```

After editing all files 

```bash
sudo apt install virtualenv
virtualenv venv --python=python3
source venv/bin/activate
sudo apt install python3-pip
pip3 install -r requirements.txt
npm install --save serverless-python-requirements 
serverless invoke local --function hello
sudo sls deploy -v
```