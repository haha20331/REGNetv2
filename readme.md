# Requirements
* GPU: NVIDIA GeForce RTX 2080 (8192 MB)
* REGNet-V2/requirements.txt 為conda安裝環境檔

# Tnference
* inference code: REGNet-V2/grasp_detect_from_file_multiobjects.py
* inference config: REGNet-V2/config.yaml
* 爪超參數: REGNet-V2/dataset_utils/eval_score/configs/config.py

# Tnput and output
## input格式: 點雲.npy (N, 6)，放在REGNet-V2/our_data
* 內部還有多個前處理code
1. depth_gen.py: 把rgbd相機取得的depth(.raw)轉為depth(.png)   --仔細想想說不定也可以直接用raw做成點雲--
2. point_npy_gen.py: 結合color.png與depth.png，做成pointcloud.npy
3. point_npy_with_mask_gen.py: 結合color.png與depth.png與Match Anything預測的object mask，做成pointcloud.npy
4. glb2np.py: 轉dust3r的pointcloud.glb成pointcloud.npy
5. vis_pc.py: 把npy的結果視覺化，需要調整相機角度

## output格式: .p，是個dict 字典，放在REGNet-V2/vis_grasp
1. ["points"]是整個場景的點雲(N, 6)
2. ["select_grasp2"]是grasp pose的SE(3) pose representation (R = T[:3, :3], T = T[:3, 3])
* example: 
    [[[-0.5088,  0.8564, -0.0870,  0.0295],
    [-0.8107, -0.5107, -0.2860, -0.3326],
    [-0.2894, -0.0749,  0.9542,  0.4817],
    [ 0.     ,  0.     ,  0.     ,  1.    ]]]
* output視覺化有兩種，結果都放在REGNet-V2/vis_grasp
1. 預測時順便畫的，但是用matplotlib，點雲與夾爪間彼此不管空間關係隨意覆蓋，不好看
2. REGNet-V2/open3d_vis_grasp.py，需要open3d套件(REGNet環境內沒裝)，畫出來就是完整點雲了，吃的輸入是REGNet output的.p

# Inference time and GPU memory
| Point Cloud Sample | GPU Memory 使用量 | 架模型時間 (s) | Grasp PoseEstimate 時間 (s)|
|-|-|-|-|
| 51200 points           | 6.27 GB            | 0.4            | 0.9  |
| 25600 points           | 4.09 GB            | 0.4            | 0.56 |