import pickle

with open("/home/tomyeh/env/REGNet-v2/REGNet-V2/test_file/results/scene1/multigrasp_layer1/0.06_analytic/camera1_1.p", "rb") as f:
    data = pickle.load(f)
    print(data.keys())        # 看有哪些欄位
    print(data["select_grasp2"].shape)
    print(data["select_grasp2"])