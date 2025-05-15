import numpy as np
import imageio
image_num = 4
# 參數
width, height = 640, 480
raw_file_path = f"./{image_num}/depth.raw"
output_png_path = f"./{image_num}/depth.png"

# 讀取 raw 檔案並轉成 2D 影像 (16-bit unsigned integer)
depth_array = np.fromfile(raw_file_path, dtype=np.uint16).reshape((height, width))

# 儲存為 depth.png
imageio.imwrite(output_png_path, depth_array)