import numpy as np
import cv2
image_num = 4
rgb = cv2.imread(f'./{image_num}/color.png')[:, :, ::-1]  # BGR to RGB
depth = cv2.imread(f'./{image_num}/depth.png', cv2.IMREAD_UNCHANGED)  # depth should be uint16 or float

fx = 366.454  # 焦距 (x 軸)
fy = 366.530  # 焦距 (y 軸)
cx = 316.605          # 主點 (中心 x)
cy = 241.725          # 主點 (中心 y)

h, w = depth.shape
i, j = np.meshgrid(np.arange(w), np.arange(h))

# 深度圖可能是 uint16（例如 Kinect2），你要轉成 meters
depth_m = depth.astype(np.float32) / 1000.0  # 假設單位為 mm

z = depth_m
x = (i - cx) * z / fx
y = (j - cy) * z / fy

points = np.stack((x, y, z), axis=2).reshape(-1, 3)
colors = rgb.reshape(-1, 3) / 255.0  # normalize RGB

# 去除深度為 0 的點
valid = (z > 0).reshape(-1)
points = points[valid]
colors = colors[valid]

pc = np.concatenate([points, colors], axis=1)  # shape (N, 6)
np.save(f"./{image_num}/pointcloud.npy", pc)