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


## Note

- The created IAM roles and policies are very powerful. Make sure to 
change and restrict them before moving to the production. 
