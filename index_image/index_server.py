import sys

sys.path.insert(1, '../Create_index')
from flask import Flask, request, Response

import jsonpickle
import numpy as np
import cv2
from nearest_neighbor import NNQuery
from extract_features import FeatureExtractor

# Initialize the Flask application
app = Flask(__name__)


@app.before_first_request
def load_dependencies():
    global nn, feature_extractor
    nn = NNQuery("../Create_index/means.npy")
    feature_extractor = FeatureExtractor(5000)


@app.route('/api/test', methods=['POST'])
# route http posts to this method
def test():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # do some fancy processing here....
    features = feature_extractor.get_descriptors(img)
    for feature in features[:500]:
        print(feature)
    visual_words = nn.find_nearest_neighbors(features)

    # build a response dict to send back to client
    response = {'message': 'Visual Words :{}'.format(visual_words)}
    # encode response using jsonpickle
    for word in visual_words[:500]:
        print(word)


    return Response(response=response, status=200, mimetype="application/json")


@app.route('/', methods=['GET'])
def hello_world():
    return "Hello World"


# start flask app

nn = None
feature_extractor = None

app.run(host="0.0.0.0", port=5000)
