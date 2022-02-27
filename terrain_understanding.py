import numpy as np
from sklearn.cluster import KMeans

class TerrainUnderstanding:
    """
    Class for terrain understanding
    """
    def __init__(self, raw_point_cloud: tuple) -> None:
        self._raw_point_cloud = raw_point_cloud
    
    def cluster(self) -> list:
        """
        Cluster the points
        :return: List of clusters
        """
        K = 4
        N = 4
        MAX_ITERS = 100
        kmeans = KMeans(n_clusters=K, 
                        max_iter=MAX_ITERS, 
                        n_init=N).fit(self._raw_point_cloud)
        return kmeans.labels_

    def findSteppableTerrain(self) -> tuple:
        """
        Find the steppable points
        :param x: x coordinates
        :param y: y coordinates
        :param z: z coordinates
        :return: Tuple of steppable points
        """
        # Cluster the points
        labels = self.cluster()
        print('labels: {}, labels shape: {}'.format(labels, np.shape(labels)))

        return labels