#!/usr/bin/env python

from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2


import rospy
import time
import numpy as np
import open3d
import pandas as pd
import matplotlib.pyplot as plt
from std_msgs.msg import String
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

hello_str = ""
pub = None
pubf = None

def talker(pub):
    rate = rospy.Rate(20) # 10hz
    while not rospy.is_shutdown():
        global hello_str
        #hello_str = "\n hello world %s" % rospy.get_time()
        if (hello_str != ""):
            rospy.loginfo(hello_str)
            pub.publish(hello_str)
            hello_str = ""
        rate.sleep()
        
def np_array_info(cloud_np):
    cloud_x = cloud_np[:, 0]
    cloud_y = cloud_np[:, 1]
    cloud_z = cloud_np[:, 2]

    x_max, x_min = np.max(cloud_x), np.min(cloud_x)
    y_max, y_min = np.max(cloud_y), np.min(cloud_y)
    z_max, z_min = np.max(cloud_z), np.min(cloud_z)

    print('x_max: ', x_max,  ', x_min: ', x_min)
    print('y_max: ', y_max, ', y_min: ', y_min)
    print('z_max: ', z_max, ', z_min: ', z_min)



def clusters(pcd):
    
    bbox = open3d.geometry.AxisAlignedBoundingBox(min_bound=(-15, -6, -3), max_bound=(15, 6, 15))

    print(bbox.get_max_bound())
    roipcd = pcd.crop(bbox)

    ## VOXEL GRID DOWNSAMPLING

    print(f"Points before downsampling: {len(roipcd.points)} ")
    vxpcd = roipcd.voxel_down_sample(voxel_size=0.25)
    print(f"Points after downsampling: {len(vxpcd.points)}") 
    t1 = time.time()

    ## SEGMENTATION WITH RANSAC

    plane_model, inliers = vxpcd.segment_plane(distance_threshold=0.3, ransac_n=3, num_iterations=150)
    inlier_cloud= vxpcd.select_by_index(inliers)
    outlier_cloud =  vxpcd.select_by_index(inliers, invert=True)
    outlier_cloud.paint_uniform_color([1,0,0])
    t2 = time.time()
    print(f"time to segment points using RANSAC {t2 - t1}")

    ## CLUSTERING WITH DBSCAN

    with open3d.utility.VerbosityContextManager(open3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(outlier_cloud.cluster_dbscan(eps=0.5, min_points=10, print_progress=False))
        
    max_label= labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    t3 = time.time()
    print(f"Time to cluster outliers using DBSCAN {t3 - t2}")

    ## 3D BOUNDING BOXES 

    
    indexes = pd.Series(range(len(labels))).groupby(labels, sort=False).apply(list).tolist()
    obbs = []
    MAX_POINTS = 500
    MIN_POINTS = 10
    global pub
    global pubf
    for i in range(0, len(indexes)):
        nb_points = len(outlier_cloud.select_by_index(indexes[i]).points)
        if (nb_points > MIN_POINTS and nb_points<MAX_POINTS):
            sub_cloud = outlier_cloud.select_by_index(indexes[i])
            obb = sub_cloud.get_axis_aligned_bounding_box()
            if(obb.get_extent()[0] < 6): # reject the bounding boxes larger than 6m
                obb.color = (1,0,0)
                rospy.loginfo(obbs)
                pub.publish("Center -> " + "000  " + "Dimensions -> " + obb.get_print_info())
                print("Box -> ")
                print(np.asarray(obb.get_box_points(), dtype=np.float32).flatten())
                pubf.publish(np.asarray(obb.get_box_points(), dtype=np.float32).flatten())
                print(obb.get_center())
                print(obb.get_print_info())
                obbs.append(obb)
    t4 = time.time()
    print(f"Number of Bounding Boxes calculated {len(obbs)}")
    print(f"Time to compute bounding boxes {t4 - t3}")
    ## VISUALIZE THE FINAL RESULTS

    #list_of_visuals = []
    #list_of_visuals.append(outlier_cloud)
    #list_of_visuals.extend(obbs)
    #list_of_visuals.append(inlier_cloud)
    #print(obbs)
    #print(type(pcd))
    #print(type(list_of_visuals))
    #open3d.visualization.draw_geometries(list_of_visuals)


def callback_pointcloud(data):
    assert isinstance(data, PointCloud2)
    points3D = point_cloud2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)
    points3D = np.array([point for point in points3D])
    print(points3D.shape)
    #np_array_info(points3D)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points3D) #convert numpy array to pointcloud
    #print(pcd.get_max_bound())
    clusters(pcd)



def main():
    global pub
    global pubf
    rospy.init_node('pcl_listener', anonymous=True)
    pub = rospy.Publisher('cluster_info', String, queue_size=100)
    pubf = rospy.Publisher('bbox3d', numpy_msg(Floats), queue_size=100)
    rospy.Subscriber('/mid/points', PointCloud2, callback_pointcloud)

    rospy.spin()

if __name__ == "__main__":
    main()
