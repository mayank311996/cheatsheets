## Run

We will use approach mentioned in this [article](https://medium.com/@mikepalei/serving-a-tensorflow-2-model-on-aws-lambda-58ce64ef7d38)
to deploy out ML model.

The reason behind this is the new TF 2.3 version will not fit under 500MB
requirement of AWS Lambda. Now, AWS got new feature that we can attach 
a EFS to our AWS Lambda functions.

So in this case, we will download all dependencies and ML model into EFS 
and then we will attach that EFS to our Lambda function. Cool!

## Resources 

- https://medium.com/@mikepalei/serving-a-tensorflow-2-model-on-aws-lambda-58ce64ef7d38
- https://aws.amazon.com/blogs/aws/new-a-shared-file-system-for-your-lambda-functions/
- https://aws.amazon.com/blogs/compute/building-deep-learning-inference-with-aws-lambda-and-amazon-efs/
- https://medium.com/@mike.p.moritz/running-tensorflow-on-aws-lambda-using-serverless-5acf20e00033
- https://stackoverflow.com/questions/62093781/run-tensorflow-2-prediction-on-aws-lambda
- https://www.edeltech.ch/tensorflow/machine-learning/serverless/2020/07/11/how-to-deploy-a-tensorflow-lite-model-on-aws-lambda.html
