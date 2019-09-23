import numpy as np


class OnlineKmeans:
    def __init__(self, n_clusters):
        self.counts = np.zeros(n_clusters, dtype=int)
        self.means = np.random.uniform(0, 255, size=(n_clusters, 128))
        self.n_clusters = n_clusters

    def reset_means(self):
        self.counts = np.zeros(self.n_clusters)
        self.means = np.random.uniform(0, 255, size=(self.n_clusters, 128))


    def set_means_with_image(self,points):
        self.means=points
        self.counts=np.zeros(self.n_clusters)

    def __find_min_distance(self, point):
        return np.argmin(np.linalg.norm(self.means - point,axis=1))

    # def __find_min_distance_batch(self, point):
    #     return np.argmin(self.means @ point.T, axis=0)

    def update_means(self, new_point):
        min_distance_cluster = self.__find_min_distance(new_point)
        print(min_distance_cluster)
        self.counts[min_distance_cluster] += 1
        self.means[min_distance_cluster] += 1 / (self.counts[min_distance_cluster]) * (
                    new_point - self.means[min_distance_cluster])

    def update_means_batch(self, new_points):
        min_distance_clusters = self.__find_min_distance_batch(new_points)

        for i, min_distance_cluster in enumerate(min_distance_clusters):
            self.counts[min_distance_cluster] += 1
            self.means[min_distance_cluster] = 1 / (self.counts[min_distance_cluster]) * (
                        new_points[i] - self.means[min_distance_cluster])

    def save_means(self):
        np.save("means.npy", self.means)
        np.save("counts.npy", self.counts)

    def load_means(self):
        self.means = np.load("means.npy")
        self.counts = np.load("counts.npy")
        self.n_clusters = self.means.shape[0]