import random
import string
import pickle


class ClusterToWord:
    cluster_word_map = "/home/adithya/Documents/reverse_image_search/es_index_image/cluster_word_map.pkl"

    def __init__(self):
        self.mapping = None

    def __check_uniqueness(self, words):
        words_set = set(words)
        return len(words_set) == len(words)

    def __generate_random_words(self, classes):
        return ["".join(random.choices(string.ascii_lowercase, k=8)) for _ in range(len(classes))]

    def create_mapping(self, classes):
        while True:
            eight_letter_words = self.__generate_random_words(classes)
            if self.__check_uniqueness(eight_letter_words):
                break
        self.mapping = dict()
        for i in range(len(classes)):
            self.mapping[classes[i]] = eight_letter_words[i]

    def get_mapping(self):
        if self.mapping is not None:
            return self.mapping
        else:
            with open(self.cluster_word_map, "rb") as f:
                self.mapping = dict()
                self.mapping = pickle.load(f)
            return self.mapping

    def dump_mapping(self):
        if self.mapping is not None:
            with open(self.cluster_word_map, "wb") as f:
                pickle.dump(self.mapping, f)
            return True
        else:
            return False

    def get_cluster_names_for_image(self, cluster_id_vector):
        if self.mapping is not None:
            return [self.mapping[i] for i in cluster_id_vector]
        else:
            with open(self.cluster_word_map, "rb") as f:
                self.mapping = dict()
                self.mapping = pickle.load(f)
                return [self.mapping[i] for i in cluster_id_vector]

