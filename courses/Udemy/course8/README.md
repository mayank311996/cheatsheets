# Exploring AWS IoT

>STEP 1
>Introduction

- Outline. Follow `1.png`
- Devices. Follow `2.png` and `3.png`
- Software used. Follow `4.png` to `8.png`
- MQTT. Follow `9.png`

>STEP 2
>Setting up AWS and some testing 

- Install AWS CLI and setup AWS configure [link](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html)
- IoT core and actions related services. Follow `10.png` to `13.png`
- IAM for IoT policies and roles. Follow `14.png` to `18.png`
- Security credentials and policies from IoT Core. Follow `19.png` to `33.png`. 
File name can be anything but extension should be `.pem` also double inverted comma 
is used to force not to have `.txt` extension in windows. Not needed for linux. 
- Communications protocols and security for devices on AWS. Follow `34.png` [link](https://aws.amazon.com/about-aws/whats-new/2018/02/aws-iot-core-now-supports-mqtt-connections-with-certificate-based-client-authentication-on-port-443/)
- Sending JSON test payloads. Follow `35.png` to `40.png` [link](https://github.com/sborsay/AWS-IoT/blob/master/AWSCLI_Payload_Tester)
- MQTT.fx tool. Follow `41.png` to `50.png` [link](https://github.com/sborsay/AWS-IoT/blob/master/AWSCLI_Payload_Tester)
[link2](https://mqttfx.jensd.de/)

>STEP 3
>RaspberryPi with AWS IoT SDK in Python 

- Provisioning AWS IoT to receive our JSON sensor data from our RaspberryPi. Follow 
`51.png` to `63.png`
- Setting up our RaspberryPi3 with the AWS SDK in Python, and the AWS CLI tool. Follow
`64.png` to `76.png`
- Modifying the basicPubSub.py program to send our data to AWS IoT with our Rpi3. Follow 
`77.png` to `87.png`

>STEP 4
>SNS

- Set up a text notification for our sensor data. Follow `88.png` to `103.png`
- Set up an email notification for our sensor data. Follow `104.png` to `116.png`
- Using conditional data testing for notifications. Follow `117.png` to `122.png`  

>STEP 5
>S3

- Saving a data object directly to S3. Follow `123.png` to `135.png`
- Exporting data to CSV or JSON. Follow `136.png` and `137.png`

>STEP 6
>Kinesis

- Introduction to Kinesis Firehose from the AWS IoT panel. Follow `138.png`
- Configuring Kinesis Firehose for data transfer. Follow `139.png` to `161.png`

>STEP 7
>DynamoDB

- Introduction to DynamoDB. Follow `162.png` and `163.png` [link](https://docs.aws.amazon.com/iot/latest/developerguide/iot-ddb-rule.html)
- Configuring the DynamoDB for our sensor data. Follow `164.png` to `185.png`

>STEP 8
>DataPipeline

- Introduction to the AWS Data Pipeline. Follow `186.png` 
- Configure and implement the Data Pipeline for data transfer to S3. 
Follow `187.png` to `199.png`

>STEP 9
>AWS Glue 

- Introduction to AWS Glue. Follow `200.png` 
- Using Glue to crawl our data file. Follow `201.png` to `213.png`
- Using a Glue ETL job to transform our JSON data to CSV. Follow `214.png`
to `225.png`

>STEP 10
>AWS QuickSight

- Introduction to AWS QuickSight. Follow `226.png` [link](https://docs.aws.amazon.com/quicksight/latest/user/getting-started-create-analysis-s3.html)
[link2](https://www.diffchecker.com/diff)
- Editing permissions and S3 bucket access. Follow `227.png` to 

## Note

- The created IAM roles and policies are very powerful. Make sure to 
change and restrict them before moving to the production. 
- Directly saving data from IoT to S3 is not good because it always saves 
one reading at a time in a separate file and overwrites the previous file that doesn't 
make any sense. So, use other approaches. 
- When you download messages from MQTT client window, it will download only the 
messages that are being displayed on the window. 
- timestamp() is AWS DynamoDB built-in function. You can look into documentation
for more built-in functions. 

