import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import scipy.linalg

class TerrainUnderstanding:
    """
    Class for terrain understanding
    """
    def __init__(self, raw_point_cloud: tuple) -> None:
        self._raw_point_cloud = raw_point_cloud

    def kmeans(self) -> list:
        """
        Cluster the points
        :return: List of labels
        """
        K = 4
        N = 4
        MAX_ITERS = 100
        kmeans = KMeans(n_clusters=K, 
                        max_iter=MAX_ITERS, 
                        n_init=N).fit(self._raw_point_cloud)
        return kmeans.labels_

    def getClusters(self, labels: list) -> list:
        """
        Get the clusters
        :param labels: List of labels
        :return: List of clusters
        """
        clusters = list()
        # Get the clusters for each label
        for label in np.unique(labels):
            label_mask = np.where(labels == label, 1, 0)
            cluster = list()
            for idx, point in enumerate(self._raw_point_cloud):
                if label_mask[idx] == 1:
                    cluster.append(point)
            clusters.append(cluster)
        return clusters

    def bestFitPlane(self, cluster: np.ndarray, order='linear') -> tuple:
        """
        Find the best fit plane
        :param cluster: Cluster of points
        :return: Tuple of plane parameters
        """
        # Convert cluster to numpy array
        cluster = np.array(cluster)
        # Find minimum x and y values
        min_x = np.min(cluster[:, 0])
        min_y = np.min(cluster[:, 1])
        # Find maximum x and y values
        max_x = np.max(cluster[:, 0])
        max_y = np.max(cluster[:, 1])

        print('min_x: {}, min_y: {}, max_x: {}, max_y: {}'.format(min_x, min_y, max_x, max_y))
        # Grid covering domain of data
        X, Y = np.meshgrid(np.arange(min_x, max_x, 0.1), np.arange(min_y, max_y, 0.1))

        # Fit plane to cluster
        if order == 'linear':
            # Best fit linear plane
            A = np.c_[cluster[:, 0], cluster[:, 1], np.ones(cluster.shape[0])]
            print('A: {}, A shape: {}'.format(A, np.shape(A)))
            # Calculate the coefficients of the plane
            C,_,_,_ = scipy.linalg.lstsq(A, cluster[:, 2])
            print('C: {}, C shape: {}'.format(C, np.shape(C)))
            Z = C[0]*X + C[1]*Y + C[2]
                
        return X, Y, Z

    def findSteppableTerrain(self) -> tuple:
        """
        Find the steppable points
        :param x: x coordinates
        :param y: y coordinates
        :param z: z coordinates
        :return: Tuple of steppable points
        """
        # Run kmeans
        labels= self.kmeans()
        print('labels: {}, labels shape: {}'.format(labels, np.shape(labels)))
        # Get the clusters
        clusters = self.getClusters(labels)
        # Plot point and fitted surface for each cluster
        for cluster in clusters:
            # Find the best fit plane
            X, Y, Z = self.bestFitPlane(cluster)
            # Convert cluster to numpy array
            cluster = np.array(cluster)
            fig = plt.figure(figsize=(10, 10))
            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, Z)
            ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], c='r', s=50)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()
        return labels