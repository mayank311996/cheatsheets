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

- Install forever and forever-service on pi. First do 
```
sudo apt-get install npm
sudo npm cache clean -f
sudo npm install -g n
sudo n stable
```
then follow `30.png`
- Install NodeJS. Follow `31.png` to `33.png`. Replace 10.x with 14.x in `31.png`
- Install AWS CLI. Follow `34.png` to `37.png`
- To install pip instead follow 
```bash
sudo apt install python3-pip
pip3 --version
python3 -m pip install --upgrade pip
```
- Configure AWS CLI. Follow `38.png`

>STEP 4
>DynamoDB

- Check `39.png`
- Create a DynamoDB table. Follow `40.png` and `41.png`
- Giving permission to Pi for DynamoDB. Follow `42.png` to `51.png`

>STEP 5
>SNS

- Check `52.png`
- Create topic. Follow `53.png` to `57.png`
- Try to send a message. Follow `58.png` to `60.png`
- Set SNS permission. Follow `61.png` to `64.png`
- Check again on Pi. Follow `65.png`

>STEP 6
>Parameter Store

- Setting up parameter. Follow `66.png` to `68.png`
- Set up permissions. Follow `69.png` to `74.png`
 
>STEP 7
>Physical Connections 

- Check `75.png`
- Set up physical connections. Follow `76.png` to `83.png`

>STEP 8
>CodeDeploy Agent

- Check `84.png`
- Set up CodeDeploy agent. Follow `85.png` to `89.png`
- Get Pi serial number. Follow `90.png` to `92.png`
- Register RaspberryPi. Follow `93.png` to `102.png`
- Check CodeDeploy dashboard. Follow `103.png`
- Delete extra permissions. Follow `104.png` and `105.png`

>STEP 9
>CodeDeploy

- Check `106.png`
- GitHub code. Follow `107.png` and `108.png`
- appspec.yml. Follow `109.png`
- Scripts. Follow `110.png` to `114.png`. 
`npm install` in `AfterInstall.sh` looks for `package.json` file 
- Cleaning home directory. Follow `115.png`
- Create CodeDeploy application. Follow `116.png` to `118.png`
- Create a deployment group. Follow `119.png` to `129.png`
- Create deployment. Follow `130.png` to `140.png`
- Change the code and deploy again. Follow `141.png` to `144.png`

>STEP 10
>CodePipeline

- Check `145.png`
- Create CodePipeline. Follow `146.png` to `155.png`
- Change some code. Follow `156.png` to `160.png`

>STEP 11
>Forever and Forever-service

- Check `161.png` and `162.png`
- Forever Hands-on. Follow `163.png` to `166.png`

>STEP 12
>app.js

- Global variables. Follow `167.png`
- Main function. Follow `168.png` 
- AWS region and DynamoDB client. Follow `169.png`
- Parameter store. Follow `170.png`
- pushButton. Follow `171.png`
- 

## Login details 

To connect `ssh app@10.0.0.155`

## Note

- Don't setup AWS on commercial RaspberryPis. Instead use AWS IoT
- Parameter store is useful to store some crucial parameters like, you don't want to include ARNs in 
your code that might be published to github. Instead you can mention in parameter store. 
- CodeDeploy can be used with thousands of RaspberryPi as well. This is good 
if you want to update your software and don't worry about updating each 
RaspberryPi. 
- In CodeDeploy you need to deploy every time manually, when change code in
GitHub repository. To avoid this we use CodePipeline, which detects changes in 
Github directly and automatically deploys code into RaspberryPi through CodeDeploy. 
- Try using Gunicorn instead of Forever. 








































