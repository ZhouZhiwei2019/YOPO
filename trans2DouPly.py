import os
import open3d as o3d

source_dir = "/home/zzw/YOPO/run/yopo_gazebo/"
target_dir = "/home/zzw/YOPO/run/yopo_gazebo_ascii/"

os.makedirs(target_dir, exist_ok=True)

for fname in os.listdir(source_dir):
    if fname.endswith(".ply"):
        print("Converting:", fname)
        pcd = o3d.io.read_point_cloud(os.path.join(source_dir, fname))
        o3d.io.write_point_cloud(os.path.join(target_dir, fname), pcd, write_ascii=True)

