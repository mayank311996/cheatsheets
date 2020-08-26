## Run
> STEP 1
#### Uploading Files to Google Cloud Storage

Upload all files in `/src` directory along with [trained model](https://www.google.com) and [vocabulary](https://www.google.com) to Google Cloud Storage Bucket.

In my case bucket is located at `gs://mayank-sentimentflaskapp/`. 

> STEP 2
#### Getting Files into Current Cloud Shell

Open the Cloud Shell and type following commands.
```
ls 
gsutil ls
gsutil ls gs://mayank-sentimentflaskapp/
gsutil cp -r gs://mayank-sentimentflaskapp/ .
ls
```
