import cv2
import time
import glob
import numpy as np


class FeatureExtractor:
    def __init__(self, n_keypoints=5000):
        self.sift = cv2.xfeatures2d.SIFT_create(n_keypoints)

    def get_descriptors_from_path(self, img_path : str):
        img = cv2.imread(img_path)
        (_, descs) = self.sift.detectAndCompute(img, None)
        return descs

    def get_descriptors(self, img : np.ndarray):
        (_, descs) = self.sift.detectAndCompute(img, None)
        return descs

def main(path):
    image_paths = glob.glob(path + "/*")
    f = FeatureExtractor(5000)
    start = time.time()
    for i, image_path in enumerate(image_paths):
        try:
            _ = f.get_descriptors(image_path)
        except:
            continue
        print(i)
    end = time.time()
    print(end - start)


if __name__ == "__main__":
    main("../sample_images")
