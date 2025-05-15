import trimesh
import numpy as np
import os
import glob

# 指定資料夾路徑
input_folder = "./dust3r"
glb_files = glob.glob(os.path.join(input_folder, "*.glb"))

for glb_path in glb_files:
    filename = os.path.splitext(os.path.basename(glb_path))[0]  # 取得檔名（不含副檔名）
    output_path = os.path.join(input_folder, f"{filename}.npy")

    scene_or_mesh = trimesh.load(glb_path)
    points = []
    colors = []

    # 處理 scene 或 mesh
    if isinstance(scene_or_mesh, trimesh.Scene):
        geometries = scene_or_mesh.geometry.values()
    else:
        geometries = [scene_or_mesh]

    for geom in geometries:
        verts = geom.vertices
        if geom.visual.kind == 'vertex' and hasattr(geom.visual, 'vertex_colors'):
            col = geom.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
        else:
            col = np.ones_like(verts) * 0.5  # 預設灰色

        points.append(verts)
        colors.append(col)

    points = np.vstack(points)
    colors = np.vstack(colors)
    pc = np.hstack((points, colors))

    np.save(output_path, pc)
    print(f"✔ 已轉換 {glb_path} → {output_path}")
