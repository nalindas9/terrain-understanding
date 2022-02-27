import matplotlib.pyplot as plt
import numpy as np

class PointCloud:
    """
    Class for point cloud processing
    """
    def __init__(self) -> None:
        self._raw_point_clouds = list()
        self._point_clouds = list()

    def loadPointCloud(self, file_path: str) -> None:
        """
        Load point cloud data from file
        :param file_path: Path to point cloud data
        :return: None
        """
        point_cloud = np.load(file_path)

        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2]

        self._point_clouds.append((x, y, z))
        self._raw_point_clouds.append(point_cloud)
        
    def plotPointCloud(self) -> None:
        """
        Plot point cloud data
        :return: None
        """
        fig = plt.figure(figsize=(10, 10))
        # Plot
        for idx, point_cloud in enumerate(self._point_clouds):
            # Add new subplot iteratively
            ax = fig.add_subplot(1, len(self._point_clouds), idx + 1, projection='3d')
            x = point_cloud[0]
            y = point_cloud[1]
            z = point_cloud[2]
            ax.scatter(x, y, z)
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')
        plt.show()

    def plotClusteredPointCloud(self, labels: list) -> None:
        """
        Plot clustered point cloud data
        :param labels: List of labels
        :return: None
        """
        fig = plt.figure(figsize=(10, 10))
        # Plot
        for idx, point_cloud in enumerate(self._point_clouds):
            # Add new subplot iteratively
            ax = fig.add_subplot(1, len(self._point_clouds), idx + 1, projection='3d')
            x = point_cloud[0]
            y = point_cloud[1]
            z = point_cloud[2]
            ax.scatter(x, y, z, c=labels[idx])
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')
        plt.show()
    
    def getPointCloud(self) -> tuple:
        """
        Get point cloud data
        :return: Tuple of point cloud data
        """
        return self._point_clouds

    def getRawPointCloud(self) -> tuple:
        """
        Get raw point cloud data
        :return: Tuple of raw point cloud data
        """
        return self._raw_point_clouds