
import requests

url = 'https://us-central1-optimal-mender-234015.cloudfunctions.net/predict_flower'
r = requests.post(url, json = {
	"sepal_length":1,
	"sepal_width":0.1,
	"petal_length":0,
	"petal_width":10
})
print(r.text)
