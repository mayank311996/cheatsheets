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
`78.png` and `79.png`
- Configuring AWS IoT Analytics to send data to a Data Lake hosted in S3.
`80.png` to `92.png`
- Setting S3 permissions, bucket policy, & CORS to allow public access to our data.
`93.png` to `106.png`
[link](https://github.com/sborsay/Serverless-IoT-on-AWS/tree/master/PublicBucket)
- Testing our Google Chart to ingest our IoT data on a remote host.
[link](https://github.com/sborsay/Serverless-IoT-on-AWS/blob/master/GoogleCharts_CSV.html)
`107.png` to `112.png`
- Moving our IoT charting webpage to S3 as a static host.
`113.png` to `122.png`

>STEP 3
>Advanced AWS IoT Analytics 

- Introduction to our advanced example. 
`123.png`
- Creating our advanced Lambda enhancement function.
[link](https://github.com/sborsay/Serverless-IoT-on-AWS/blob/master/Advanced_IoTAnalytics/transformLambda.js)
`124.png` to `137.png`
- Testing our Lambda enhancement connecting it to IoT Analytics.
[link](https://github.com/sborsay/Serverless-IoT-on-AWS/blob/master/Advanced_IoTAnalytics/test_payload.json)
`138.png` to `157.png`
- The Arduino Sketch to send GPS Coordinates via MQTT and filling our S3 bucket.
[link](https://github.com/sborsay/Serverless-IoT-on-AWS/blob/master/Advanced_IoTAnalytics/GPS_Sketch_ESP32.ino)
[link2](https://github.com/sborsay/Serverless-IoT-on-AWS/blob/master/Advanced_IoTAnalytics/GPS_Sketch_ESP8266.ino)
`158.png` to `169.png`
- Reviewing our IoT design flow thus far, and discussing next steps.
`170.png` 
- Connecting our second Lambda to extract CSV IoT data from our S3 data bucket.
[link](https://github.com/sborsay/Serverless-IoT-on-AWS/blob/master/Advanced_IoTAnalytics/extract_csv-data-Lambda.js)
`171.png` to `180.png`
- Creating a REST API endpoint with AWS API Gateway to our extraction Lambda.
`181.png` to `197.png`
- Creating our website in S3 to visualize our IoT Analytics data in Highcharts.
`198.png` to `213.png`
[link](https://github.com/sborsay/Serverless-IoT-on-AWS/blob/master/Advanced_IoTAnalytics/index.html)
- Preview: Adding Security to your visualization web sight.
[link](https://github.com/sborsay/Serverless-IoT-on-AWS/blob/master/Advanced_IoTAnalytics/API_Key_Secured_index.html)

>STEP 4
>Level One Serverless IoT for data lakes, using IoT Core, Lambda, and S3

- Levels of a Serveless design flow for IoT data.
`214.png` and `215.png`
- Intro to ingestion methods.
`216.png`
- IoT Core to S3 using Lambda part 1.
[link](https://github.com/sborsay/Serverless-IoT-on-AWS/blob/master/S3lambdaManyFolders.js)
[link2](https://github.com/sborsay/Serverless-IoT-on-AWS/tree/master/PublicBucket)
`217.png` to `224.png`
- IoT Core to S3 using Lambda part 2.
[link](https://github.com/sborsay/Serverless-IoT-on-AWS/blob/master/ESP8266-to-AWS-IoT)
[link2](https://github.com/sborsay/Serverless-IoT-on-AWS/blob/master/ESP32-to-AWS-IoT)
`225.png` to `234.png`
- IoT Core to S3 using Lambda part 3. 
[link](https://github.com/sborsay/Serverless-IoT-on-AWS/blob/master/myS3lambda.js)
`235.png` to ``