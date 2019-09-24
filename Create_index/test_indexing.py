from clustering import OnlineKmeans, OfflineKmeans
from extract_features import FeatureExtractor
import time
import glob


def main(path):
    online_clusterer = OnlineKmeans(5000)
    # offline_clusterer = OfflineKmeans(5000)
    online_clusterer.load_means()
    image_paths = glob.glob(path + "/*")
    f = FeatureExtractor(5000)
    start = time.time()
    # descriptors = []
    for i, image_path in enumerate(image_paths):
        try:
            descriptors = f.get_descriptors_from_path(image_path)
        except:
            print("error")
            continue

        for descriptor in descriptors:
            online_clusterer.update_means(descriptor)

    online_clusterer.save_means()
    end = time.time()
    print(end - start)


if __name__ == "__main__":
    main("../sample_images")
