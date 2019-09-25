from __future__ import print_function
import requests
import json
import cv2

addr = 'http://localhost:5000'
test_url = addr + '/api/test'

image_id = 'test_1.JPEG'
img = cv2.imread(image_id)
print(img)

# encode image as jpeg
_, img_encoded = cv2.imencode('.JPEG', img)

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}
params = {'image_id': image_id}

# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tostring(), headers=headers, params=params)
# decode response

print(response.__dict__)