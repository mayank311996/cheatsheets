# Home Automation with Raspberry Pi and AWS - IoT - 2019

## Steps to setup RaspberryPi

>STEP 1
>Setting up RaspberryPi 

- Download raspberry [image](https://www.raspberrypi.org/downloads/raspberry-pi-os/) 
- Copy to SD card using https://www.balena.io/etcher/ tool
- Go to `boot` directory and create one file named `ssh` without any extension
- Download `wpa_supplicant.conf` file and edit details with country code, wifi id, and wifi password. Then copy it 
to the `boot` directory
- Eject the flash card from laptop 
- Insert SD card into RaspberryPi and power it on 
- Find RaspberryPi PublicIP through your wifi and SSH into it. 
Default id = `pi` and password = `raspberry`
- Add new user as per `8.png` and `9.png`
- Giving user default SUDO power without typing password. Follow `10.png`
and `11.png`
- Granting permission for GPIO (input/output). Follow `12.png`
- Testing sudo power of newly added user `app`. Follow `13.png`
- Now logout and check if you can login with use `app`, like `ssh app@xxx.xxx.xxx.xxx`. 
Follow `14.png`
- Delete user `pi`. Follow `15.png`
- Try ssh using user `pi`. You should get access denied message as we deleted the user already 
- Update and Upgrade. Follow `16.png`
- Create key pair for RaspberryPi. Follow `17.png` to `23.png`. After that login using 
public and private key through Putty. This step is not needed if you don't mind typing password all the time.
Linux also has similar procedure. Check on the internet
- If you did follow previous step then follow this one too. Removing access via id and password.
Follow `24.png` to `26.png`

>STEP 2
>Setting up AWS

- Create new IAM user named `pi`. Follow `27.png` to `29.png`. Also, download and
store credentials at secure place

>STEP 3

- Install forever and forever-service on pi. Follow `30.png`
- Install NodeJS. Follow `31.png` to `33.png`
- Install AWS CLI. Follow `34.png` to `37.png`
- Configure AWS CLI. Follow `38.png`

>STEP 4
>DynamoDB

- Check `39.png`
- 

## Login details 

To connect `ssh app@10.0.0.155`

## Note

- Don't setup AWS on commercial RaspberryPis. Instead use AWS IoT
