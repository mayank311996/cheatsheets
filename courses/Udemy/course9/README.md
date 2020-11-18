# AWS Serverless Design for IoT

>STEP 1 
>Introduction 

- Check `1.png` to `5.png` [link](https://aws.amazon.com/about-aws/whats-new/2018/02/aws-iot-core-now-supports-mqtt-connections-with-certificate-based-client-authentication-on-port-443/)
[link2](https://github.com/sborsay/AWS-IoT/blob/master/AWSCLI_Payload_Tester)

>STEP 2 
>AWS IoT Analytics 

- Introduction to AWS IoT Analytics. Follow `6.png` and `7.png` [link](https://docs.aws.amazon.com/iotanalytics/)
[link2](https://docs.aws.amazon.com/iotanalytics/latest/userguide/quickstart.html)
- Configuring AWS IoT analytics channel, pipeline, and datastore. 
`8.png` to `17.png`
- Configuring our Arduino sketch to send sensor JSON data package to AWS IoT Core.
[link](https://github.com/sborsay/Serverless-IoT-on-AWS/blob/master/simPub_ESP8266.ino)
[link2](https://github.com/sborsay/Serverless-IoT-on-AWS/blob/master/simPub_ESP32.ino)
`18.png` to `20.png`
- Discussing Lambda and using Lambda enhancement in IoT Analytics.
`21.png` to `29.png`
- Hands-on with lambda and testing a Lambda function with a test data payload.
`30.png` to `37.png`
- Enhancing our incoming IoT data in Lambda.
`38.png` to `60.png`
[link](https://github.com/sborsay/Serverless-IoT-on-AWS/blob/master/myNode_IoT_Lambda)
- Using AWS QuickSight with our data produced from AWS IoT Analytics. 
`61.png` to `67.png`
- Using AWS SageMaker with our data produced from AWS IoT Analytics.
[link](https://github.com/sborsay/AWS-IoT/blob/master/AWS_Pandas_Sagemaker.py)
`68.png` to `77.png`
- Why hosting a Data Lake may be superior to just invoking our Data Set.
`78.png` to 
