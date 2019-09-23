from clustering import OnlineKmeans
from extract_features import FeatureExtractor
import time
import glob


def main(path):
    clusterer = OnlineKmeans(5000)

    image_paths = glob.glob(path + "/*")
    f = FeatureExtractor(5000)
    start = time.time()
    initial_seed = f.get_descriptors_from_path(image_paths[0])
    clusterer.set_means_with_image(initial_seed)

    for i, image_path in enumerate(image_paths):
        try:
            descriptors = f.get_descriptors_from_path(image_path)
        except:
            print("error")
            continue

        for descriptor in descriptors:

            clusterer.update_means(descriptor)
        print(i)
        del descriptors

    end = time.time()
    clusterer.save_means()
    print(end - start)


if __name__=="__main__":
    main("../sample_images")