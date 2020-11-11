# Home Automation with Raspberry Pi and AWS - IoT - 2019

## Steps to setup RaspberryPi

- Download raspberry [image](https://www.raspberrypi.org/downloads/raspberry-pi-os/) 
- Copy to SD card using https://www.balena.io/etcher/ tool
- Go to `boot` directory and create one file named `ssh` without any extension
- Download `wpa_supplicant.conf` file and edit details with country code, wifi id, and wifi password. Then copy it 
to the `boot` directory
- Eject the flash card from laptop 
- Insert SD card into RaspberryPi and power it on 
- Find RaspberryPi PublicIP through your wifi and SSH into it
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
