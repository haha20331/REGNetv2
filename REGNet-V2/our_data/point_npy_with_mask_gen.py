import numpy as np
import cv2

image_num = 3
# 1. 讀入 RGB, Depth, Mask
rgb = cv2.imread(f'./{image_num}/color.png')[:, :, ::-1]  # BGR to RGB
depth = cv2.imread(f'./{image_num}/depth.png', cv2.IMREAD_UNCHANGED)
mask = cv2.imread(f'./{image_num}/object_mask.png', cv2.IMREAD_UNCHANGED)  # ❗ 0/1 mask，不轉灰階！

# 2. 相機內參
fx = 366.454
fy = 366.530
cx = 316.605
cy = 241.725

# 3. 轉換座標
h, w = depth.shape
i, j = np.meshgrid(np.arange(w), np.arange(h))

depth_m = depth.astype(np.float32) / 1000.0  # mm → m
z = depth_m
x = (i - cx) * z / fx
y = (j - cy) * z / fy

points = np.stack((x, y, z), axis=2).reshape(-1, 3)
colors = rgb.reshape(-1, 3) / 255.0
mask_flat = (mask.reshape(-1) > 128) 


valid = (z > 0).reshape(-1) & mask_flat

points = points[valid]
colors = colors[valid]

pc = np.concatenate([points, colors], axis=1)
np.save(f"./{image_num}/our{image_num}_mask.npy", pc)