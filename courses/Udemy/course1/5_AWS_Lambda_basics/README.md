## Run
> STEP 1
#### Installing serverless

```
curl -o- -L https://slss.io/install | bash
```

> STEP 2
#### To check the installation

```
sls
```

> STEP 3
#### Configure the serverless

```
sls config --provider aws --key 'key_from_AWS' --secret 'secret_key_from_AWS'
```

> STEP 4
#### To create a severless project

```
mkdir demo_sls
cd demo_sls/
sls create --template aws-python3 --name 'name'
```
