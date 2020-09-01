## Run

> STEP 1
#### Creating environment for spaCy

```bash
conda create -n spacy-dev python=3.6 pylint rope jupyter
conda activate spacy-dev
pip install spacy
python -m spacy download en_core_web_sm
```

> STEP 2 
#### Serverless project

```bash
sls create --template aws-python3 --name ner-api
sls plugin install -n serverless-python-requirements@4.2.4
```