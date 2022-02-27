import numpy as np
import matplotlib.pyplot as plt
from point_cloud import PointCloud
from terrain_understanding import TerrainUnderstanding

if __name__ == '__main__':
    # Load point cloud data
    point_cloud = PointCloud()
    point_cloud.loadPointCloud('dataset/grid_stairs.npy')
    point_cloud.loadPointCloud('dataset/grid_water.npy')
    point_cloud.loadPointCloud('dataset/curb_w_grass.npy')
    # Plot point cloud data
    point_cloud.plotPointCloud()
    # Get raw point cloud data
    raw_point_clouds = point_cloud.getRawPointCloud()
    all_labels = list()
    
    # For each point cloud, find steppeable points
    for idx, raw_point_cloud in enumerate(raw_point_clouds):
        # Create terrain understanding object
        terrain_understanding = TerrainUnderstanding(raw_point_cloud)
        labels, clusters = terrain_understanding.findSteppableTerrain()
        all_labels.append(labels)
        # Plot Steppable Terrain
        terrain_understanding.plotSteppableTerrain(clusters=clusters)

    # Plot clustered point cloud data
    point_cloud.plotClusteredPointCloud(all_labels)