import sys

sys.path.insert(1, '../Create_index')
sys.path.append('../es_index_image')

import cv2
from nearest_neighbor import NNQuery
from extract_features import FeatureExtractor
import cluster_to_word
import dao
import glob
import os

IMAGE_DATASET_PATH = "../../tiny-imagenet-200/train/*/*/*.JPEG"
nn = None
feature_extractor = None

es_dao = dao.EsDao('localhost')


def assign_index(clustered_image_vector, image_id):
    es = cluster_to_word.ClusterToWord()
    image_cluster_list = es.get_cluster_names_for_image(clustered_image_vector)
    clustered_image = " ".join(image_cluster_list)
    return clustered_image


def extract_features(img):
    # do some fancy processing here....
    features = feature_extractor.get_descriptors(img)
    # for feature in features[:500]:
    #     print(feature)
    if features is None:
        return None
    clustered_image_vector = nn.find_nearest_neighbors(features)
    return clustered_image_vector


def main():
    global nn, feature_extractor
    nn = NNQuery("../Create_index/means.npy")
    feature_extractor = FeatureExtractor(5000)
    image_paths = glob.glob(IMAGE_DATASET_PATH)
    docs = dict()
    for counter, image_path in enumerate(image_paths):
        image_id = os.path.basename(image_path)
        img = cv2.imread(image_path)
        clustered_image_vector = extract_features(img)
        if clustered_image_vector is None:
            print("could not extract features for image {}".format(image_id))
            continue
        clustered_image = assign_index(clustered_image_vector, image_id)
        docs[image_id] = clustered_image
        if (counter+1) % 1000 == 0:
            print(es_dao.store_in_bulk(docs), counter)
            docs = dict()


if __name__ == '__main__':
    main()
