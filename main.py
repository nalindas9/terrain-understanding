import numpy as np
import matplotlib.pyplot as plt
from point_cloud import PointCloud
from terrain_understanding import TerrainUnderstanding

if __name__ == '__main__':
    # Load point cloud data
    point_cloud = PointCloud()
    #point_cloud.loadPointCloud('dataset/grid_stairs.npy')
    #point_cloud.loadPointCloud('dataset/grid_water.npy')
    point_cloud.loadPointCloud('dataset/curb_w_grass.npy')
    # Plot point cloud data
    point_cloud.plotPointCloud()
    # # Get point cloud data
    # point_clouds = point_cloud.getPointCloud()
    # # Create terrain understanding object
    # terrain_understanding = TerrainUnderstanding()
    # # For each point cloud, find steppeable points
    # for point_cloud in point_clouds:
    #     x = point_cloud[0]
    #     y = point_cloud[1]
    #     z = point_cloud[2]
    #     steppable_points = terrain_understanding.steppable(x, y, z)

    # Get raw point cloud data
    raw_point_clouds = point_cloud.getRawPointCloud()
    print('Raw point cloud shape: {}'.format(np.shape(raw_point_clouds)))
    # For each point cloud, find steppeable points
    for raw_point_cloud in raw_point_clouds:
        print('Raw point cloud: {}'.format(raw_point_cloud))
        all_labels = list()
        # Create terrain understanding object
        terrain_understanding = TerrainUnderstanding(raw_point_cloud)
        labels = terrain_understanding.findSteppableTerrain()
        all_labels.append(labels)

    # Plot clustered point cloud data
    point_cloud.plotClusteredPointCloud(all_labels)