## Run
> STEP 1
#### Installing a serverless plugin

```
sls plugin install -n serverless-python-requirements@4.2.4
```

> STEP 2
#### Creating and deploying the serverless project

```
sls create --template aws-python3 --name diabetes_prediction_final
sls deploy
```

use postman to make request to the end point 

