
## IMPORT LIBRARIES
import numpy as np
import time
import open3d
import pandas as pd
import matplotlib.pyplot as plt

## OPEN A FILE AND VISUALIZE THE POINT CLOUD
# The supported extension names are: pcd, ply, xyz, xyzrgb, xyzn, pts.

inputPath = 'pcd_data/0000000001.pcd'

pcd = open3d.io.read_point_cloud(inputPath)

#open3d.visualization.draw_geometries([pcd])

bbox = open3d.geometry.AxisAlignedBoundingBox(min_bound=(-15, -6, -3), max_bound=(15, 6, 15))

print(bbox.get_max_bound())
roipcd = pcd.crop(bbox)

## VOXEL GRID DOWNSAMPLING

print(f"Points before downsampling: {len(roipcd.points)} ")
vxpcd = roipcd.voxel_down_sample(voxel_size=0.25)
print(f"Points after downsampling: {len(vxpcd.points)}") 
#open3d.visualization.draw_geometries([vxpcd])
t1 = time.time()

## SEGMENTATION WITH RANSAC

plane_model, inliers = vxpcd.segment_plane(distance_threshold=0.3, ransac_n=3, num_iterations=150)
inlier_cloud= vxpcd.select_by_index(inliers)
outlier_cloud =  vxpcd.select_by_index(inliers, invert=True)
outlier_cloud.paint_uniform_color([1,0,0])
t2 = time.time()
print(f"time to segment points using RANSAC {t2 - t1}")
#open3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

## CLUSTERING WITH DBSCAN

with open3d.utility.VerbosityContextManager(open3d.utility.VerbosityLevel.Debug) as cm:
	labels = np.array(outlier_cloud.cluster_dbscan(eps=0.5, min_points=10, print_progress=False))
	
max_label= labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
outlier_cloud.colors = open3d.utility.Vector3dVector(colors[:, :3])
colors[labels < 0] = 0
outlier_cloud.colors = open3d.utility.Vector3dVector(colors[:, :3])
t3 = time.time()
print(f"Time to cluster outliers using DBSCAN {t3 - t2}")
#open3d.visualization.draw_geometries([outlier_cloud])
t3 = time.time()

## 3D BOUNDING BOXES 

obbs = []
indexes = pd.Series(range(len(labels))).groupby(labels, sort=False).apply(list).tolist()

MAX_POINTS = 500
MIN_POINTS = 10
for i in range(0, len(indexes)):
    nb_points = len(outlier_cloud.select_by_index(indexes[i]).points)
    if (nb_points > MIN_POINTS and nb_points<MAX_POINTS):
        sub_cloud = outlier_cloud.select_by_index(indexes[i])
        obb = sub_cloud.get_axis_aligned_bounding_box()
        if(obb.get_extent()[0] < 6): # reject the bounding boxes larger than 6m
            obb.color = (1,0,0)
            obbs.append(obb)
print(f"Number of Bounding Boxes calculated {len(obbs)}")

## VISUALIZE THE FINAL RESULTS

list_of_visuals = []
list_of_visuals.append(outlier_cloud)
list_of_visuals.extend(obbs)
list_of_visuals.append(inlier_cloud)

t4 = time.time()
print(type(pcd))
print(type(list_of_visuals))
print(f"Time to compute bounding boxes {t4 - t3}")
open3d.visualization.draw_geometries(list_of_visuals)