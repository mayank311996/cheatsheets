## Run

```bash
conda create -n scikit-dev python=3.6
conda activate scikit-dev
conda install scikit-learn=0.20.2 jupyter pandas pylint rope
conda list 
```

```bash
sls create --template aws-python3 --name california-housing
```

Installing on plugin
```bash
sls plugin install -n serverless-python-requirements@4.2.4 
```

- When you have backend and frontend on different domains 
you need to specify "Access-Control-Allow-Origin": "*" in 
handler.py. For now this means we allow any domain to query our 
backend but in production you should specify only one domain that can 
query your backend.
