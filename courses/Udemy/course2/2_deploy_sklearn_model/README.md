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
sls plugin install -n serverless-python-requirements 
```