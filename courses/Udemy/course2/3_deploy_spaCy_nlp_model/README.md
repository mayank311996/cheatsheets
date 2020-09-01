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

- In this case we are zipping requirements as well. So, it is zip inside
another zip. This is to reduce the size of the package. And to unzip
the requirements when Lambda functions cold starts we need try and except
block in the beginning. The import unzip_requirements comes from 
sls plugin we installed. 