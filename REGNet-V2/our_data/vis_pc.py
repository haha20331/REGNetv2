import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from mpl_toolkits.mplot3d import Axes3D

# 指定資料夾
input_folder = "./dust3r/central_np"
npy_files = glob.glob(os.path.join(input_folder, "*.npy"))

for npy_path in npy_files:
    folder = os.path.dirname(npy_path)
    filename = os.path.splitext(os.path.basename(npy_path))[0] 
    save_path = folder + "/vis/" + filename + ".png"
    pc = np.load(npy_path)
    if len(pc) > 25600:
        select_point_index = np.random.choice(len(pc), 25600, replace=False)
        pc = pc[select_point_index]

    points = pc[:, :3]
    colors = pc[:, 3:6]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=0.5)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=-90, azim=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✔ 已儲存：{save_path}")
