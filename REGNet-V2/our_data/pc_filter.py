import numpy as np
import os
import glob
from sklearn.cluster import DBSCAN

# 設定搜尋路徑
npy_files = glob.glob("./dust3r/*.npy")

for npy_path in npy_files:
    folder = os.path.dirname(npy_path)
    filename = os.path.splitext(os.path.basename(npy_path))[0] 
    save_path = folder + "/central_np/" + filename + "_central.np"
    pc = np.load(npy_path)
    xyz = pc[:, :3]

    # 使用 DBSCAN 找密集區
    clustering = DBSCAN(eps=0.02, min_samples=10).fit(xyz)
    labels = clustering.labels_

    # 找出最大的 cluster（排除 label = -1）
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(unique) == 0:
        print(f"⚠ 無法從 {npy_path} 找到密集區，略過。")
        continue
    densest_label = unique[np.argmax(counts)]
    dense_points = xyz[labels == densest_label]

    # 以該密集區的平均作為中心
    dense_center = dense_points.mean(axis=0)

    # 範圍過濾
    radius = 0.12  # 可調整半徑大小
    dists = np.linalg.norm(xyz - dense_center, axis=1)
    mask = dists < radius
    pc_central = pc[mask]

    # 儲存
    
    np.save(save_path, pc_central)

    print(f"✔ {npy_path}: 原始點數 {len(pc)}, 中心點數 {len(pc_central)}")

# import numpy as np

# # 載入 [N,6] 的點雲資料
# pc = np.load("./mike/pointcloud.npy")

# # 取出空間座標
# xyz = pc[:, :3]

# # 計算空間中心
# center = xyz.mean(axis=0)

# # 計算每個點到中心的距離
# dists = np.linalg.norm(xyz - center, axis=1)

# # 設定保留距離中心的半徑，例如 0.5（你可以調整這個值）
# radius = 0.1
# mask = dists < radius

# # 過濾點雲
# pc_central = pc[mask]

# # 儲存結果
# np.save("./mike/pointcloud_central.npy", pc_central)

# print(f"原始點數: {pc.shape[0]}，保留中心點數: {pc_central.shape[0]}")
