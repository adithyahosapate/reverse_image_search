import sys

sys.path.insert(1, '../Create_index')
sys.path.append('../es_index_image')
from flask import Flask, request, Response

import numpy as np
import cv2
from nearest_neighbor import NNQuery
from extract_features import FeatureExtractor
import cluster_to_word
import dao

# Initialize the Flask application
app = Flask(__name__)


@app.before_first_request
def load_dependencies():
    global nn, feature_extractor
    nn = NNQuery("../Create_index/means.npy")
    feature_extractor = FeatureExtractor(5000)


def index_and_store(clustered_image_vector, image_id):
    es = cluster_to_word.ClusterToWord()
    image_cluster_list = es.get_cluster_names_for_image(clustered_image_vector)
    clustered_image = " ".join(image_cluster_list)
    es_dao = dao.EsDao('localhost')
    return es_dao.store_clustered_image(image_id, clustered_image)


@app.route('/api/test', methods=['POST'])
# route http posts to this method
def test():
    r = request
    image_id = r.args.get('image_id')
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # do some fancy processing here....
    features = feature_extractor.get_descriptors(img)
    # for feature in features[:500]:
    #     print(feature)
    clustered_image_vector = nn.find_nearest_neighbors(features)
    image_store_flag = index_and_store(clustered_image_vector, image_id)

    # build a response dict to send back to client
    response = {'message': 'stored image successfully={}'.format(image_store_flag)}
    # # encode response using jsonpickle
    # for word in visual_words[:500]:
    #     print(word)
    return Response(response=response, status=200, mimetype="application/json")


@app.route('/', methods=['GET'])
def hello_world():
    return "Hello World"


# start flask app

nn = None
feature_extractor = None

app.run(host="0.0.0.0", port=5000)
