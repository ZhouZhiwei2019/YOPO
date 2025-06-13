import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ruamel.yaml import YAML
import time
from scipy.spatial.transform import Rotation as R


class YopoDataset(Dataset):
    def __init__(self):
        super(YopoDataset, self).__init__()
        cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] + "/flightlib/configs/traj_opt.yaml", 'r'))
        scale = 32  # 神经网络下采样倍数，图像分辨率除以这个
        self.height = scale * cfg["vertical_num"]
        self.width = scale * cfg["horizon_num"]
        multiple_ = 0.5 * cfg["vel_max"]
        # The x-direction follows a log-normal distribution,
        # while the yz-direction follows a normal distribution with a mean of 0.
        self.v_max = cfg["vel_max"]
        v_des = multiple_ * cfg["vx_mean_unit"]
        self.vx_lognorm_mean = np.log(self.v_max - v_des)
        self.vx_logmorm_sigma = np.log(np.sqrt(v_des))
        self.v_mean = multiple_ * np.array([cfg["vx_mean_unit"], cfg["vy_mean_unit"], cfg["vz_mean_unit"]])
        self.v_var = multiple_ * multiple_ * np.array([cfg["vx_var_unit"], cfg["vy_var_unit"], cfg["vz_var_unit"]])
        self.a_mean = multiple_ * multiple_ * np.array([cfg["ax_mean_unit"], cfg["ay_mean_unit"], cfg["az_mean_unit"]])
        self.a_var = multiple_ * multiple_ * multiple_ * multiple_ * np.array([cfg["ax_var_unit"], cfg["ay_var_unit"], cfg["az_var_unit"]])

        print("Indexing dataset (image paths and labels)...")
        data_cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] + "/flightlib/configs/vec_env.yaml", 'r'))
        data_dir = os.environ["FLIGHTMARE_PATH"] + data_cfg["env"]["dataset_path"]

        self.img_list = []
        self.map_idx = []
        self.positions = np.empty((0, 3))
        self.quaternions = np.empty((0, 4))
        subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
        subfolders.sort(key=lambda x: int(os.path.basename(x)))
        for i, folder in enumerate(subfolders):
            print(f"[{i}] Indexing folder: {folder}")
            file_names = [f for f in os.listdir(folder) if f.endswith(".tif")]
            file_names.sort(key=lambda x: int(x.split('.')[0].split("_")[1]))
            full_paths = [os.path.join(folder, fname) for fname in file_names]
            self.img_list.extend(full_paths)
            self.map_idx.extend([i] * len(full_paths))

            label_path = os.path.join(folder, "label.npz")
            labels = np.load(label_path)
            self.positions = np.vstack((self.positions, labels["positions"]))
            self.quaternions = np.vstack((self.quaternions, labels["quaternions"]))

        print(f"Dataset indexed: {len(self.img_list)} images.")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_path = self.img_list[item]
        img = cv2.imread(img_path, -1).astype(np.float32)

        # Resize if needed
        if img.shape[-2] != self.height or img.shape[-1] != self.width:
            img = cv2.resize(img, (self.width, self.height))
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)

        vel, acc = self._get_random_state()

        q_wxyz = self.quaternions[item]
        R_WB = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        euler_angles = R_WB.as_euler('ZYX', degrees=False)
        R_wB = R.from_euler('ZYX', [0, euler_angles[1], euler_angles[2]], degrees=False)
        goal_w = np.random.randn(3) + np.array([2, 0, 0])
        goal_b = R_wB.inv().apply(goal_w)

        goal_dist = np.linalg.norm(goal_b)
        goal_dir = goal_b / goal_dist
        random_obs = np.hstack((vel, acc, goal_dir))

        return img, self.positions[item], self.quaternions[item], random_obs, self.map_idx[item]

    def _get_random_state(self):
        vel = self.v_mean + np.sqrt(self.v_var) * np.random.randn(3)
        acc = self.a_mean + np.sqrt(self.a_var) * np.random.randn(3)

        right_skewed_vx = -1
        while right_skewed_vx < 0:
            right_skewed_vx = np.random.lognormal(mean=self.vx_lognorm_mean, sigma=self.vx_logmorm_sigma)
            right_skewed_vx = -right_skewed_vx + self.v_max + 0.2
        vel[0] = right_skewed_vx
        return vel, acc


if __name__ == '__main__':
    dataset = YopoDataset()
    data_loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=32,
        pin_memory=True,
        prefetch_factor=8,
        persistent_workers=True
    )

    start = time.time()
    for i, (depth, pos, quat, obs, idx) in enumerate(data_loader):
        print(f"Batch {i} loaded: shape={depth.shape}")
        if i >= 5:
            break
    end = time.time()
    print("样本加载测试完成，总耗时：", end - start)
