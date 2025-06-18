#!/usr/bin/env python3
import rospy
import tf
import numpy as np
import open3d as o3d
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import os
import cv2

# ---------- 参数 ----------
MAX_FRAMES = 500                      # 最大帧数
SAVE_PATH = "/home/zzw/YOPO/run/yopo_gazebo/map1.ply"
DEPTH_TOPIC = "/iris0/camera/depth/image_raw" 
CAMERA_INFO_TOPIC = "/iris0/camera/depth/camera_info"
ODOM_TOPIC = "/drone0/mavros/local_position/odom"
FRAME_ID = "map"
VOXEL_DOWNSAMPLE = 0.05

# ---------- 相机参数 ----------
camera_intrinsics = None  # Will be filled in callback
bridge = CvBridge()
pcd = o3d.geometry.PointCloud()
frame_count = 0


def get_camera_intrinsic_from_msg(msg):
    K = msg.K
    fx, fy = K[0], K[4]
    cx, cy = K[2], K[5]
    width = msg.width
    height = msg.height
    return fx, fy, cx, cy, width, height


def depth_to_point_cloud(depth_image, pose, fx, fy, cx, cy, scale=1.0):
    h, w = depth_image.shape
    i_range = np.arange(h)
    j_range = np.arange(w)
    jj, ii = np.meshgrid(j_range, i_range)
    z = depth_image * scale
    x = (jj - cx) * z / fx
    y = (ii - cy) * z / fy
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    valid = (z > 0.1) & (z < 14.9)
    points = points[valid.reshape(-1)]
    ones = np.ones((points.shape[0], 1))
    points_homo = np.concatenate([points, ones], axis=1).T  # [4, N]

    # --- 相机坐标 → 机体坐标（FLU） ---
    R = np.array([
        [0,  0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ])
    T_cam_to_body = np.eye(4)
    T_cam_to_body[:3, :3] = R
    T_cam_to_body[0, 3] = 0.15  # 相机前移 0.15m

    points_body = T_cam_to_body @ points_homo

    # --- 机体坐标 → 世界坐标 ---
    T_body_to_world = np.eye(4)
    T_body_to_world[:3, :3] = tf.transformations.quaternion_matrix(pose["quat"])[:3, :3]
    T_body_to_world[:3, 3] = pose["pos"]

    points_world = (T_body_to_world @ points_body)[:3].T
    return points_world

def callback(depth_msg, caminfo_msg, odom_msg):
    global camera_intrinsics, frame_count, pcd

    if camera_intrinsics is None:
        fx, fy, cx, cy, w, h = get_camera_intrinsic_from_msg(caminfo_msg)
        camera_intrinsics = (fx, fy, cx, cy)

    try:
        encoding = depth_msg.encoding
        if encoding == "32FC1":
            depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
        elif encoding == "16UC1":
            depth_raw = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
            depth = depth_raw.astype(np.float32) * 0.001
        else:
            rospy.logerr(f"[Depth] Unsupported encoding: {encoding}")
            return
    except Exception as e:
        rospy.logerr(f"Failed to convert depth image: {e}")
        return


    pose = {
        "pos": np.array([odom_msg.pose.pose.position.x,
                         odom_msg.pose.pose.position.y,
                         odom_msg.pose.pose.position.z]),
        "quat": np.array([odom_msg.pose.pose.orientation.x,
                          odom_msg.pose.pose.orientation.y,
                          odom_msg.pose.pose.orientation.z,
                          odom_msg.pose.pose.orientation.w])  # xyzw
    }

    fx, fy, cx, cy = camera_intrinsics
    cloud = depth_to_point_cloud(depth, pose, fx, fy, cx, cy)
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud)
    pcd += cloud_o3d
    frame_count += 1

    print(f"[Frame {frame_count}] Current total points: {len(pcd.points)}")

    if frame_count >= MAX_FRAMES:
        print(f"Saving accumulated point cloud to {SAVE_PATH}")
        pcd = pcd.voxel_down_sample(VOXEL_DOWNSAMPLE)
        o3d.io.write_point_cloud(SAVE_PATH, pcd)
        print("Done!")
        rospy.signal_shutdown("Finished collection.")


if __name__ == "__main__":
    rospy.init_node("gazebo_depthmap_to_ply", anonymous=False)
    depth_sub = message_filters.Subscriber(DEPTH_TOPIC, Image)
    caminfo_sub = message_filters.Subscriber(CAMERA_INFO_TOPIC, CameraInfo)
    odom_sub = message_filters.Subscriber(ODOM_TOPIC, Odometry)

    ts = message_filters.ApproximateTimeSynchronizer([depth_sub, caminfo_sub, odom_sub],
                                                     queue_size=10, slop=0.1)
    ts.registerCallback(callback)

    rospy.loginfo("Started collecting pointcloud from Gazebo depth + odom...")
    rospy.spin()

