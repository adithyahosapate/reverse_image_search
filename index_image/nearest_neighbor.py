from scipy.spatial import KDTree
import numpy as np

class NNQuery:

    def __init__(self, means:np.ndarray):
        self.means= means
        self.KDTree = KDTree(means)

    def __init__(self, means_path:str):
        self.means = np.load(means_path)
        print( self.means)
        self.KDTree = KDTree(self.means)

    def find_nearest_neighbors(self, points):
        return [np.argmin(np.linalg.norm(self.means - point,axis=0)) for point in points]


    def reset_points(self,means):
        self.means=means
        self.KDTree = KDTree(means)