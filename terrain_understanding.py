import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import scipy.linalg
from matplotlib import cm
import richdem as rd

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
        K = 6
        N = 6
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
        X, Y = np.meshgrid(np.arange(min_x, max_x, 0.1), np.arange(min_y, max_y+0.1, 0.1))
        X_flatten = X.flatten()
        Y_flatten = Y.flatten()
        # Fit plane to cluster
        if order == 'linear':
            # Best fit linear plane
            A = np.c_[cluster[:, 0], cluster[:, 1], np.ones(cluster.shape[0])]
            #print('A: {}, A shape: {}'.format(A, np.shape(A)))
            # Calculate the coefficients of the plane
            C,_,_,_ = scipy.linalg.lstsq(A, cluster[:, 2])
            #print('C: {}, C shape: {}'.format(C, np.shape(C)))
            Z = C[0]*X + C[1]*Y + C[2]
        elif order == 'quadratic':
            # Best fit quadratic plane
            A = np.c_[np.ones(cluster.shape[0]), cluster[:, :2], np.prod(cluster[:, :2], axis=1), cluster[:, :2]**2]
            #print('A: {}, A shape: {}'.format(A, np.shape(A)))
            # Calculate the coefficients of the plane
            C,_,_,_ = scipy.linalg.lstsq(A, cluster[:, 2])
            #print('C: {}, C shape: {}'.format(C, np.shape(C)))
            Z = np.dot(np.c_[np.ones(X_flatten.shape), X_flatten, Y_flatten, X_flatten*Y_flatten, X_flatten**2, Y_flatten**2], C).reshape(X.shape)
                
        return X, Y, Z

    def checkFlatSurface(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> bool:
        """
        Check if the surface is flat
        :param clusters: List of clusters
        :return: True if flat, False otherwise
        """
        # Calculate gradient
        Z_flatten = Z.flatten()
        grad_z = np.gradient(Z_flatten)
        print('grad_z: {}, grad_z shape: {}'.format(grad_z, np.shape(grad_z)))
        # Check if the magnitude of the gradient is less than 0.1
        print('np.max(np.abs(grad_z)):', np.max(np.abs(grad_z)))
        if np.max(np.abs(grad_z)) < 0.02:
            return True
        else:
            return False

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
        #print('labels: {}, labels shape: {}'.format(labels, np.shape(labels)))
        # Get the clusters
        clusters = self.getClusters(labels)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection='3d')
        # Plot point and fitted surface for each cluster
        for cluster in clusters:
            # Find the best fit plane
            X, Y, Z = self.bestFitPlane(cluster, order='quadratic')
            print('X: {}, X shape: {}'.format(X, np.shape(X)))
            print('Y: {}, Y shape: {}'.format(Y, np.shape(Y)))
            print('Z: {}, Z shape: {}'.format(Z, np.shape(Z)))
            # Convert cluster to numpy array
            cluster = np.array(cluster)
            self.checkFlatSurface(X, Y, Z)
            # Check if the surface is flat
            if self.checkFlatSurface(X, Y, Z):
                # Plot the cluster
                ax.plot_surface(X, Y, Z, color='green')
                ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], c='b', s=20)
            else:
                ax.plot_surface(X, Y, Z, color='red')
                ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], c='b', s=20)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        plt.show()
        return labels