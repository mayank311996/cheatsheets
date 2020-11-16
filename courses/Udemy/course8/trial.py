import time
import random
import json
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

# For certificate based connection
myMQTTClient = AWSIoTMQTTClient("FGC_RaspberryPi")
# For TLS mutual authentication
myMQTTClient.configureEndpoint(
    "a21rin7oh3q2pr-ats.iot.us-east-2.amazonaws.com",
    8883
)  # Provide your AWS IoT Core endpoint (
# Example: "abcdef12345-ats.iot.us-east-1.amazonaws.com")
myMQTTClient.configureCredentials(
    "/home/app/certificates/AmazonRootCA1.pem",
    "/home/app/certificates/7f35a63383-private.pem.key",
    "/home/app/certificates/7f35a63383-certificate.pem.crt"
)  # Set path for Root
# CA and unique device credentials (use the private key and certificate
# retrieved from the logs in Step 1)
myMQTTClient.configureOfflinePublishQueueing(-1)
myMQTTClient.configureDrainingFrequency(2)
myMQTTClient.configureConnectDisconnectTimeout(10)
myMQTTClient.configureMQTTOperationTimeout(5)

print("Initializing IoT core topic...")
myMQTTClient.connect()

# Publish gps coordinates to AWS IoT Core

# print("Publishing message from FGC_RaspberryPi")
# myMQTTClient.publish(
#     topic="home/trial",
#     QoS=1,
#     payload="{'Message': 'Ping from FGC_RaspberryPi'}"
# )

topic = "home/trial"
loopCount = 0
while True:
    message = {}
    message['message'] = 'Message from FGCRaspberryPi'
    message['sequence'] = loopCount
    messageJson = json.dumps(message)
    python_object = {
        'Device_ID': 'FGC RaspberryPi4',
        'time': time.time(),
        'Temperature': random.randint(0, 100),
        'Humidity': random.randint(0, 50)
    }
    json_string = json.dumps(python_object)
    myAWSIoTMQTTClient.publish(topic, json_string, 1)

    print('Published topic %s: %s\n' % (topic, json_string))
    loopCount += 1
    time.sleep(10)
