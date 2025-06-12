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
from scipy.spatial.transform import Rotation

# ---------- 参数 ----------
TOTAL_GROUPS = 52
FRAMES_PER_GROUP = 2048
IMG_WIDTH = 640
IMG_HEIGHT = 480
DEPTH_TOPIC = "/iris0/camera/depth/image_raw/new"
CAMERA_INFO_TOPIC = "/iris0/camera/depth/camera_info"
ODOM_TOPIC = "/drone0/mavros/local_position/odom"
SAVE_BASE = "/home/zzw/YOPO/run/yopo_gazebo"
VOXEL_DOWNSAMPLE = 0.15

# ---------- 初始化变量 ----------
camera_intrinsics = None
bridge = CvBridge()
frame_id = 0
group_id = 50
pcd = o3d.geometry.PointCloud()
positions = []
quaternions = []

# 创建保存目录
def get_group_dir(group_id):
    path = os.path.join(SAVE_BASE, str(group_id))
    os.makedirs(path, exist_ok=True)
    return path

def get_camera_intrinsic_from_msg(msg):
    K = msg.K
    fx, fy = K[0], K[4]
    cx, cy = K[2], K[5]
    return fx, fy, cx, cy

def depth_to_point_cloud(depth_image, pose, fx, fy, cx, cy, scale=1.0):
    h, w = depth_image.shape
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_image * scale
    x = (jj - cx) * z / fx
    y = (ii - cy) * z / fy
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    valid = (z > 0.1) & (z < 7.9)
    points = points[valid.reshape(-1)]
    ones = np.ones((points.shape[0], 1))
    points_homo = np.concatenate([points, ones], axis=1).T  # [4, N]

    # 相机 -> 机体 FLU
    R = np.array([[0,  0, 1], [-1, 0, 0], [0, -1, 0]])
    T_cam_to_body = np.eye(4)
    T_cam_to_body[:3, :3] = R
    T_cam_to_body[0, 3] = 0.15
    points_body = T_cam_to_body @ points_homo

    # 机体 -> world
    T_body_to_world = np.eye(4)
    T_body_to_world[:3, :3] = tf.transformations.quaternion_matrix(pose["quat"])[0:3, 0:3]
    T_body_to_world[:3, 3] = pose["pos"]

    points_world = (T_body_to_world @ points_body)[:3].T
    return points_world

def callback(depth_msg, caminfo_msg, odom_msg):
    global camera_intrinsics, frame_id, group_id, pcd, positions, quaternions

    if group_id >= TOTAL_GROUPS:
        rospy.signal_shutdown("Finished all data groups.")
        return

    if camera_intrinsics is None:
        camera_intrinsics = get_camera_intrinsic_from_msg(caminfo_msg)

    try:
        if depth_msg.encoding == "32FC1":
            depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
        elif depth_msg.encoding == "16UC1":
            depth_raw = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
            depth = depth_raw.astype(np.float32) * 0.001
        else:
            rospy.logerr(f"[Depth] Unsupported encoding: {depth_msg.encoding}")
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
                          odom_msg.pose.pose.orientation.w])
    }

    # --- 处理异常深度值 ---
    invalid_mask = np.isnan(depth) | (depth < 0.1)
    depth[invalid_mask] = 0.0

    # --- 归一化保存为.tif图像 ---
    normalized = np.clip(depth / 14.9, 0.0, 1.0)
    normalized[depth == 0.0] = 0.0
    depth_8u = (normalized * 255).astype(np.uint8)

    # 保存图像
    group_dir = get_group_dir(group_id)
    img_path = os.path.join(group_dir, f"image_{frame_id}.tif")
    cv2.imwrite(img_path, depth_8u)

    # 存pose
    positions.append(pose["pos"])
    quat = pose["quat"]
    rot = Rotation.from_quat(quat)
    q = rot.as_quat()
    quaternions.append(q)

    # 点云累计
    fx, fy, cx, cy = camera_intrinsics
    cloud = depth_to_point_cloud(depth, pose, fx, fy, cx, cy)
    cloud = cloud.astype(np.float32)  # 保持float32，后面写文件时转换
    cloud_o3d = o3d.geometry.PointCloud()
    # 这里只能传float64给 Vector3dVector，先转换下
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud.astype(np.float64))

    pcd += cloud_o3d

    print(f"[Group {group_id}] Frame {frame_id}: Total Points: {len(pcd.points)}")
    frame_id += 1

    if frame_id >= FRAMES_PER_GROUP:
        # 保存label
        np.savez(os.path.join(group_dir, "label.npz"),
                 positions=np.array(positions),
                 quaternions=np.array(quaternions))
        # 保存ply
        pcd = pcd.voxel_down_sample(VOXEL_DOWNSAMPLE)

        # 转换点云坐标为 float32 写入ply
        pcd_save = o3d.geometry.PointCloud()
        pcd_save.points = o3d.utility.Vector3dVector(np.asarray(pcd.points).astype(np.float32))
        ply_path = os.path.join(SAVE_BASE, f"pointcloud-{group_id}.ply")
        o3d.io.write_point_cloud(ply_path, pcd_save, write_ascii=True)

        print(f"[Group {group_id}] Saved to {ply_path}")

        # 清空数据，进入下一个group
        group_id += 1
        frame_id = 0
        pcd.clear()
        positions.clear()
        quaternions.clear()

if __name__ == "__main__":
    rospy.init_node("getGloPly_combined", anonymous=False)
    depth_sub = message_filters.Subscriber(DEPTH_TOPIC, Image)
    caminfo_sub = message_filters.Subscriber(CAMERA_INFO_TOPIC, CameraInfo)
    odom_sub = message_filters.Subscriber(ODOM_TOPIC, Odometry)

    ts = message_filters.ApproximateTimeSynchronizer([depth_sub, caminfo_sub, odom_sub],
                                                     queue_size=10, slop=0.1)
    ts.registerCallback(callback)

    rospy.loginfo("[getGloPly_combined] Starting data collection...")
    rospy.spin()