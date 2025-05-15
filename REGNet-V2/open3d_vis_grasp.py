import open3d as o3d
import numpy as np
import pickle
import argparse
import yaml
import glob
import os
from scipy.spatial.transform import Rotation
# 環境不能在regnet上跑，REGNet環境沒奘完整open3d
# ====== 參數 ======
object = "dust3r/usb_cable2"
exp = "layer2"
gp_path = f"./vis_grasp/{object}/{exp}/"
OUTPUT_IMAGE_PATH = gp_path + "op3d_vis.png"

# ====== 載入 grasp 資料 ======
p_files = glob.glob(os.path.join(gp_path, "*.p"))
with open(p_files[0], "rb") as f:
    grasp_info = pickle.load(f)

points = grasp_info["points"].squeeze()
xyz = points[:, :3]
colors = points[:, 3:] if points.shape[1] > 3 else None

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
if colors is not None:
    pcd.colors = o3d.utility.Vector3dVector(colors)

# ====== 載入 config.yaml ======
def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=gp_path + 'config.yaml')
    cli_args = parser.parse_args()
    with open(cli_args.config, 'r') as f:
        config = yaml.safe_load(f)
    return argparse.Namespace(**config)

args = load_config()

# ====== 建立夾爪模型 ======
def create_box(center, size, rot_mat, color):
    box = o3d.geometry.TriangleMesh.create_box(width=size[0], height=size[1], depth=size[2])
    box.paint_uniform_color(color)
    box.translate(-box.get_center())
    vertices = np.asarray(box.vertices)
    vertices -= np.mean(vertices, axis=0)
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.rotate(rot_mat, center=(0, 0, 0))
    box.translate(center)
    return box

def draw_real_gripper(center, R, base_color_offset=0.0):
    W = 0.050
    D = 0.0455
    T = 0.0082
    H = 0.024
    offset_y = W / 2
    offset_x = D / 2 + T / 2

    gripper_parts = []
    base_color = [0.5 + base_color_offset, 0.5, 0.5]
    left_color = [0.0, 1.0 - base_color_offset, 0.0]
    right_color = [1.0 - base_color_offset, 0.0, 0.0]

    gripper_parts.append(create_box(center, size=(T, W, H), rot_mat=R, color=base_color))
    left_center = center + R @ np.array([offset_x, offset_y, 0])
    right_center = center + R @ np.array([offset_x, -offset_y, 0])
    gripper_parts.append(create_box(left_center, size=(D, T, H), rot_mat=R, color=left_color))
    gripper_parts.append(create_box(right_center, size=(D, T, H), rot_mat=R, color=right_color))
    return gripper_parts

# ====== 組裝場景 ======
geometries = [pcd]

# 使用 select_grasp2 畫出正確姿勢的夾爪
g = grasp_info["select_grasp2"]  # shape: (3, 4)
R = g[:, :3]
center = g[:, 3]
gripper = draw_real_gripper(center, R)
geometries.extend(gripper)

# ====== 渲染設定 ======
w, h = 800, 600
render = o3d.visualization.rendering.OffscreenRenderer(w, h)
scene = render.scene
scene.set_background([1, 1, 1, 1])

for geo in geometries:
    scene.add_geometry(f"obj_{id(geo)}", geo, o3d.visualization.rendering.MaterialRecord())

# ====== 設定相機位置（固定從上往下）=====
def setup_camera_lookat_center(render, geometries, euler_deg, distance=0.3):
    all_points = np.vstack([np.asarray(pcd.points)] + [
        np.asarray(g.vertices) for g in geometries if isinstance(g, o3d.geometry.TriangleMesh)
    ])
    center = all_points.mean(axis=0)

    # 不用 euler_deg 計算 forward，用固定從上往下
    up = np.array([0, 0, 1])
    eye = center + np.array([-0.4, -0.3, 0.2])

    render.setup_camera(60.0, center, eye, up)

setup_camera_lookat_center(render, geometries, args.camera_transform["euler_deg"])

# ====== 渲染並儲存圖片 ======
img = render.render_to_image()
o3d.io.write_image(OUTPUT_IMAGE_PATH, img)
print(f"✅ 成功儲存圖片：{OUTPUT_IMAGE_PATH}")
