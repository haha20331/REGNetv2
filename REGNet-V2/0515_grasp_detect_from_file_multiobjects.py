#!/usr/bin/env python

import argparse
import os, copy
import time
import pickle
import open3d
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import transforms3d
import matplotlib.pyplot as plt
import yaml
import shutil

import utils_add as utils
import contrast.model_utils as contrast_utils
from contrast.tdgpd.models.build_model import build_model 
from contrast.tdgpd.utils.file_logger import file_logger_noselect as file_logger_cls
from contrast.tdgpd.yacs_config import load_cfg_from_file
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    cli_args = parser.parse_args()

    # 讀取 YAML
    with open(cli_args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 可以把 dict 轉成 Namespace，如果你想用 args.xxx 風格
    return argparse.Namespace(**config)

args = load_config()
args.cuda = args.cuda if torch.cuda.is_available else False

np.random.seed(int(time.time()))
if args.cuda:
    torch.cuda.manual_seed(1)

class GraspDetector:
    def __init__(self, method='multigrasp_layer1', visualization=False):
        self.gpu_num = args.gpu
        torch.cuda.set_device(self.gpu_num)
        self.method = args.method
        self.all_points_num = args.all_points_num
        # self.all_points_num = 51200
        ######## 幹你娘這個要改
        ######## /REGNet-V2/dataset_utils/eval_score/eval_utils/evaluation_data_generator.py
        ######## 上面路徑是碰撞測試的code
        self.table_height = args.table_height
        if "gpd" in self.method:
            self.all_points_num = 51200
        
        # self.bounds = [-0.6, 0.6, -0.6, 0]
        ######## 這是他媽的workspace，也要改，不然就隨照片定義
        self.bounds = args.bounds


        ########## 這她媽不知道三小，但感覺也要改，操你媽這個參數根本沒用到
        # self.center_camera = np.array([-0.1023, -0.0864, 1.6796])
        # self.rot_x_angle = 0.2*np.pi#-0.87*np.pi
        
        self.model, self.eval_machine = None, None
        self._init_model()
        self._init_evaluator()
        self.eval_params    = [self.depth, self.width, self.table_height, self.gpu_num]

        self.detect_num = 0
        print("Construct network successfully!")


    def _init_model(self):
        obj_class_num = 43
        gripper_num = 64

        self.width, self.height, self.depth = args.eval_width, 0.010, 0.06
        grasp_channel = 9
        radius = 0.06
        max_radius = 0.10
                
        self.gripper_params = [self.width, self.height, self.depth]

        if 'multigrasp' in self.method or 'regnet' == self.method:
            if self.method == 'multigrasp_layer1':
                sample_layer=0
                conf_times=0.03
                multi_flag = True
            elif self.method == 'multigrasp_layer2':
                sample_layer=1
                conf_times=0.15
                multi_flag = True
            elif self.method == 'multigrasp_layer3':
                sample_layer=2
                conf_times=0.75
                multi_flag = True
            elif self.method == 'regnet':
                sample_layer=0
                conf_times=0.0025
                multi_flag = False

            model_params   = [grasp_channel, radius, max_radius, obj_class_num, gripper_num, 1, self.gpu_num, '', 0.005]
            model_machine = utils.ModelInit('test', args.load_model, 'class_01', model_params, \
                                            rgb_flag=True, multi_flag=multi_flag, sample_layer=sample_layer, \
                                            conf_times=conf_times, use_region=True, use_fps=False)
            self.model, _, _, _ = model_machine.construct_model()
        
        elif self.method == 'pointnetgpd':
            grasp_points_num = 1000
            point_channel = 3
            class_num     = 2
            model_params   = [1, self.gpu_num, '', 0.005]
            self.gripper_params = self.gripper_params + [grasp_points_num]
            model_machine = contrast_utils.ModelInit('test', args.load_model, model_params, grasp_points_num, point_channel, class_num)
            self.model, _, _, _ = model_machine.construct_model()

        elif 'gpd' in self.method:
            grasp_points_num = 1000
            close_region_points_num = 1000
            project_chann = int(self.method.split('gpd')[-1])
            model_params   = [1, self.gpu_num, '', 0.005]
            project_chann = 3 if '3' in self.method else 12
            self.gripper_params = self.gripper_params + [close_region_points_num, project_chann]
            model_machine = contrast_utils.GPDModelInit('test', args.load_model, model_params, grasp_points_num, project_chann)
            self.model, _, _, _ = model_machine.construct_model()
        
        elif self.method == 'random':
            pass

        elif self.method == 's4g':
            cfg = load_cfg_from_file('/data1/cxg6/Multigrasp/contrast/configs/curvature_model.yaml')
            self.model, _, _ = build_model(cfg)

            checkpoint = torch.load(args.load_model, map_location='cuda:{}'.format(self.gpu_num))
            new_model_dict = {}
            model_dict = checkpoint['net']

            for key in model_dict.keys():
                new_model_dict[key.replace("module.", "")] = model_dict[key]
            self.model.load_state_dict(new_model_dict)
            if self.gpu_num != -1:
                self.model = self.model.cuda()
        self.model = self.model.eval() if self.model is not None else None
        torch.set_grad_enabled(False)

        
    def _init_evaluator(self):
        topK=1000
        if 'multigrasp' in self.method or 'regnet' == self.method or self.method == 's4g':
            self.eval_machine   = utils.EvalNoTruth(topK)
        elif 'gpd' in self.method or self.method=='random':
            center_num = 2000 # sample points in *gpd and random method
            self.eval_machine   = contrast_utils.EvalNoTruth(center_num, topK)
    
    def get_view_init(self):
        euler_deg = args.camera_transform["euler_deg"]  # XYZ 歐拉角 (deg)
        euler_rad = np.radians(euler_deg)  # 轉成 rad
        R = transforms3d.euler.euler2mat(*euler_rad, axes='sxyz')
        view_dir = R[:, 2]  # 第三列是相機 z 軸在世界中的方向
        # Step 4: 根據視線方向算 elev, azim
        x, y, z = view_dir
        azim = np.degrees(np.arctan2(y, x))   # xy 平面角度
        elev = np.degrees(np.arcsin(z / np.linalg.norm(view_dir)))
        return -elev, (azim+180)%360

    def draw_box(self, center, size, rot_mat, color='red', alpha=1):
        dx, dy, dz = size[0] / 2, size[1] / 2, size[2] / 2
        corners = np.array([
            [-dx, -dy, -dz],
            [ dx, -dy, -dz],
            [ dx,  dy, -dz],
            [-dx,  dy, -dz],
            [-dx, -dy,  dz],
            [ dx, -dy,  dz],
            [ dx,  dy,  dz],
            [-dx,  dy,  dz]
        ])
        corners = (rot_mat @ corners.T).T + center

        faces = [
            [0,1,2,3],
            [4,5,6,7],
            [0,1,5,4],
            [2,3,7,6],
            [1,2,6,5],
            [3,0,4,7]
        ]

        face_verts = [[corners[idx] for idx in face] for face in faces]
        return Poly3DCollection(face_verts, facecolors=color, edgecolors='k', linewidths=0.5, alpha=alpha)
    
    def draw_real_gripper(self, center, R):
        """
        繪製真實尺寸夾爪（依據 W=50mm, D=45.5mm, T=8.2mm, H=24mm）
        """
        boxes = []

        # 單位轉換為公尺
        W = 0.050
        D = 0.0455
        T = 0.0082
        H = 0.024

        offset_y = W / 2
        offset_x = D / 2 + T / 2

        # 底部支架 (灰色)
        base_center = center
        boxes.append(self.draw_box(base_center, size=(T, W, H), rot_mat=R, color='gray'))

        # 左手指 (綠色)
        left_center = center + R @ np.array([offset_x, offset_y, 0])
        boxes.append(self.draw_box(left_center, size=(D, T, H), rot_mat=R, color='green'))

        # 右手指 (紅色)
        right_center = center + R @ np.array([offset_x, -offset_y, 0])
        boxes.append(self.draw_box(right_center, size=(D, T, H), rot_mat=R, color='red'))
        return boxes
    
    def save_grasp_image_matplotlib(self, pc, grasp, save_path="grasp_result", index=0):
        print(grasp[:20])
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        all_boxes = []  # 所有夾爪幾何物件
        if grasp.ndim == 2 and grasp.shape == (4, 4):
            grasp = grasp[np.newaxis, :, :]
        
        for i, g in enumerate(grasp):
            if not np.all(np.isfinite(g)):
                continue
            center = g[:3, 3]
            R = g[:3, :3]
            all_boxes.extend(self.draw_real_gripper(center, R))

        for box in all_boxes:
            ax.add_collection3d(box)
        # 點雲
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=pc[:, 3:6], s=1, alpha=0.1)
        
        cx, cy, cz = grasp[0][:3, 3]
        ax.set_xlim(cx - 0.5, cx + 0.5)
        ax.set_ylim(cy - 0.5, cy + 0.5)
        ax.set_zlim(cz - 0.2, cz + 0.35)
        ax.set_box_aspect([1, 1, 1])
        # ax.view_init(elev=10, azim=90) # 原始demo的參數
        # ax.view_init(elev=80, azim=130)  # our1的參數
        # ax.view_init(elev=50, azim=80)  # our2
        # ax.view_init(elev=45, azim=105)
        init_elev, init_azim = self.get_view_init()
        ax.dist = 5
        camera_view = save_path + "camera_view.png"
        horizontal_view = save_path + "horizontal_view.png"

        if args.vis_surround:
            for angle in range(0, 360, 10):
                ax.view_init(elev=args.horizontal_view_elev, azim=angle)
                plt.tight_layout()
                plt.savefig(camera_view[:-4]+f"_{angle}.png", dpi=300)
        else:
            ax.view_init(elev=init_elev, azim=init_azim)
            # plt.tight_layout()
            plt.savefig(camera_view, dpi=300)
            ax.view_init(elev=args.horizontal_view_elev, azim=args.horizontal_view_azim)
            # plt.tight_layout()
            plt.savefig(horizontal_view, dpi=300)

        plt.close()
        print(f"[✓] 成功儲存完整夾爪圖，共繪製 {len(grasp)} 個 grasp → {camera_view}")

    def to_torch(self, pc):
        pc_torch = torch.Tensor(pc).view(1, -1, 6)
        cur_idx = torch.arange(len(pc_torch))
        data_width = torch.tensor([args.eval_width])

        if self.gpu_num != -1:
            pc_torch = pc_torch.cuda()
            cur_idx = cur_idx.cuda()
            data_width = data_width.cuda()
        return pc_torch, cur_idx, data_width

    def _heurestic(self, s_a, s_v):
        return s_a+s_v

    def _analytic(self, s_a, s_v):
        y = (0.8783*s_a-0.0587)/(1+np.exp(-10.1244*(s_v-0.6103)))
        return y

    def method_multigrasp(self, pc):
        '''
         pc: [N,6]
        '''
        final_grasp = np.zeros((4,4))
        pc_torch, cur_idx, data_width = self.to_torch(pc)
        # pc_mean = pc_torch.mean(dim=1)
        ## pre2_grasp: [B, N*anchor_number, 9], pre1_grasp: [B, N*anchor_number, 9]
        if self.method == 'regnet':
            pre2_grasp, pre1_grasp, _, _, _ = self.model(pc_torch, self.gripper_params, None, None, None, cur_idx)
        else:
            pre2_grasp, pre1_grasp, _, _, _ = self.model(pc_torch, self.gripper_params, None, None, None, cur_idx, data_width)
        if len(pre2_grasp) >= 1:
            select_grasp, keep_idx = self.eval_machine.eval_notruth(pc_torch, pre2_grasp, self.eval_params)
            keep_grasp = select_grasp[0][keep_idx[0].to(select_grasp[0].device)]
            print(f"keep_grasp nums: {len(keep_grasp)}")
            if len(keep_grasp) > 0:
                grasp_mat, score_antipodal, score_center = self.inv_transform_grasp(keep_grasp)
                print(f"grasp_mat nums: {len(grasp_mat)}")
                if len(grasp_mat) > 0:
                    score_vertical = np.asarray(self.com_z_score_torch(grasp_mat))
                    print(f"score_antipodal: {score_antipodal.max()}, score_vertical: {score_vertical.max}")

                    
                    y1 = self._analytic(score_antipodal,score_vertical)
                    y2 = self._heurestic(score_antipodal,score_vertical)
                    sort_index1 = np.argsort(y1)[::-1][0] # the first
                    sort_index2 = np.argsort(y2)[::-1][0] # the first
                    # print(score_antipodal[sort_index], score_vertical[sort_index])
                    final_grasp1  = grasp_mat.cpu().numpy()[sort_index1]
                    final_grasp2  = grasp_mat.cpu().numpy()[sort_index2]
                    grasp_num = min(len(grasp_mat), args.vis_grasp_nums)
                    sort_index20 = np.argsort(y1)[::-1][:grasp_num] 
                    final_grasp_top20 = grasp_mat.cpu().numpy()[sort_index20]
                    pad = np.tile(np.array([0, 0, 0, 1]).reshape(1, 1, 4), (grasp_num, 1, 1)) 
                    final_grasp_20 = np.concatenate([final_grasp_top20, pad], axis=1)

                    grasp_save_path = os.path.abspath( self.base_path + f"/{args.exp_name}/grasp_info.p")
                    if not os.path.exists(self.base_path + f"/{args.exp_name}"):
                        os.makedirs(self.base_path + f"/{args.exp_name}")
                    saved_score = np.array([score_antipodal[sort_index2], score_vertical[sort_index2], \
                                            score_antipodal[sort_index1], score_vertical[sort_index1]])
                    data = {
                        'points': pc_torch.cpu().numpy(),
                        'grasps': pre2_grasp.cpu().numpy(),
                        'keep_grasps': keep_grasp.cpu().numpy(),
                        'select_grasp':final_grasp2,  # heurestic
                        'select_grasp2':final_grasp1, # analytic
                        'score': saved_score
                    }
                    with open(grasp_save_path, 'wb') as file:
                        pickle.dump(data, file)

                    # print ('antipadol: {}, angle {}, vertical {}'.format(score_antipodal[sort_index],\
                    #     score_center[sort_index], score_vertical[sort_index]) )
                    final_grasp = np.r_[final_grasp1, np.array([0,0,0,1]).reshape(1,4)]
        return final_grasp, final_grasp_20

    def method_s4g(self, pc):
        '''
         pc: [N,6]
        '''
        final_grasp = np.zeros((4,4))
        pc_torch = torch.Tensor(pc).view(1, -1, 6)
        if self.gpu_num != -1:
            pc_torch = pc_torch.cuda()

        data_batch = {'scene_points': torch.tensor(pc[:,:3].T.reshape(1,3,-1)).cuda().float()}
        preds = self.model(data_batch)
        grasps = file_logger_cls(data_batch, preds, 0, '', '', with_label=False, gpu_id=self.gpu_num)
        
        if len(grasps) >= 1:
            select_grasp, keep_idx = self.eval_machine.eval_notruth(pc_torch, grasps, self.eval_params)
            keep_grasp = select_grasp[0][keep_idx[0]]
            if len(keep_grasp) > 0:
                grasp_mat, score_antipodal, score_center = self.inv_transform_grasp(keep_grasp)
                if len(grasp_mat) > 0:
                    score_vertical = np.asarray(self.com_z_score_torch(grasp_mat))
                    print(score_antipodal.max(), score_vertical.max())

                    if args.use_analytic:
                        y = self._analytic(score_antipodal,score_vertical)
                    else:
                        y = self._heurestic(score_antipodal,score_vertical)
                    sort_index = np.argsort(y)[::-1][0]#+score_center)[::-1][0] # the first
                    print(score_antipodal[sort_index], score_vertical[sort_index])
                    final_grasp  = grasp_mat.cpu().numpy()[sort_index]

                    grasp_save_path = os.path.abspath(self.base_path+"{}/{}/{}_1.p".format(self.method, self.width, self.pc_name))
                    if not os.path.exists(self.base_path+"{}/{}".format(self.method, self.width)):
                        os.makedirs(self.base_path+"{}/{}".format(self.method, self.width))

                    data = {
                        'points': pc_torch.cpu().numpy(),
                        'grasps': grasps.cpu().numpy(),
                        'keep_grasps': keep_grasp.cpu().numpy(),
                        'select_grasp':final_grasp,
                    }
                    with open(grasp_save_path, 'wb') as file:
                        pickle.dump(data, file)

                    final_grasp = np.r_[final_grasp, np.array([0,0,0,1]).reshape(1,4)]
        return final_grasp

    def method_gpd(self, pc):
        '''
         pc: [N,6]
        '''
        final_grasp = np.zeros((4,4))
        pc_torch = torch.Tensor(pc).view(1, -1, 6)
        if self.gpu_num != -1:
            pc_torch = pc_torch.cuda()
        grasp_mat = self.eval_machine.eval_notruth_returngrasp(self.model, pc_torch, self.eval_params, self.gripper_params)
        print(grasp_mat.shape)

        if len(grasp_mat) > 0:
            keep_indx = torch.arange(len(grasp_mat))
            keep_indx = (grasp_mat[:,0,3] < self.bounds[1]-0.1) & (grasp_mat[:,0,3] > self.bounds[0]+0.1) & \
                        (grasp_mat[:,1,3] < self.bounds[3]-0.1) & (grasp_mat[:,1,3] > self.bounds[2]+0.1)
            if keep_indx.sum()>0:
                grasp_mat = grasp_mat[keep_indx]
                score_vertical = np.asarray(self.com_z_score_torch(grasp_mat))
                print("score_vertical.max ", score_vertical.max())
                sort_index = np.argsort(score_vertical)[::-1][0] # the first
                final_grasp = grasp_mat[sort_index].cpu().numpy()

                grasp_save_path = os.path.abspath(self.base_path+"{}/{}/{}_1.p".format(self.method, self.width, self.pc_name))
                if not os.path.exists(self.base_path+"{}/{}".format(self.method, self.width)):
                    os.makedirs(self.base_path+"{}/{}".format(self.method, self.width))

                data = {
                    'points': pc_torch.cpu().numpy(),
                    'keep_grasps_mat': grasp_mat.cpu().numpy(),
                    'select_grasp_mat':final_grasp,
                }
                with open(grasp_save_path, 'wb') as file:
                    pickle.dump(data, file)
        return final_grasp

    def generate_grasp_from_file(self, pc, pc_path):
        print("######################################################")
        #np.random.seed(1)
        # pc: [N, 6] -> [x y z r g b]
        start = time.time()
        pc_paths = pc_path.split('/')[1:-1]
        pc_paths = [i+'/results' if i=='test_file' else i for i in pc_paths]
        self.base_path  = args.vis_output
        self.pc_name = pc_path.split('/')[-1].split(".npy")[0] \
            if '.npy' in pc_path else pc_path.split('/')[-1].split(".p")[0]

        torch.cuda.set_device(self.gpu_num)
        torch.set_grad_enabled(False)

        pc_formal = pc.copy()
        pc, show_pc = self._pretreat_pc(pc, real_data=True)
        # show_pc = self._get_show_pc(pc_formal, real_data)
        pc_xyz = pc[:,:3]
        # grasp [4,4] numpy
        init_time = time.time()
        print("init time = ", init_time - start)
        if 'multigrasp' in self.method or 'regnet' == self.method:
            grasp, grasp5 = self.method_multigrasp(pc)
        elif 'gpd' in self.method or self.method=='random':
            grasp = self.method_gpd(pc)
        elif self.method == 's4g':
            grasp = self.method_s4g(pc)
        final_grasp = np.matmul(np.linalg.inv(self._local_to_global_transformation_mat()), \
                                    grasp)[:3]
        # if 'multigrasp' in self.method or 'regnet' == self.method:
        #     strategy = 'analytic' if args.use_analytic else 'heuristic'
        #     grasp_save_path = os.path.abspath(self.base_path+"{}/{}_{}/{}_0.p".format(self.method, self.width, strategy, self.pc_name))
        #     if not os.path.exists(self.base_path+"{}/{}_{}".format(self.method, self.width, strategy)):
        #         os.makedirs(self.base_path+"{}/{}_{}".format(self.method, self.width, strategy))
        # else:
        #     grasp_save_path = os.path.abspath(self.base_path+"{}/{}/{}_0.p".format(self.method, self.width, self.pc_name))
        #     if not os.path.exists(self.base_path+"/{}/{}".format(self.method, self.width)):
        #         os.makedirs(self.base_path+"{}/{}".format(self.method, self.width))
        # data = {
        #     'points': pc_formal,
        #     'grasp': final_grasp
        # }
        infer_time = time.time()
        print("inference time = ", infer_time - init_time)
        self.detect_num += 1
        # with open(grasp_save_path, 'wb') as file:
        #     pickle.dump(data, file)
        vis_save_path = args.vis_output + args.exp_name + '/'
        if args.vis_lot_grasp:
            self.save_grasp_image_matplotlib(show_pc, grasp5, vis_save_path, self.detect_num)  # 畫前20個高分grasp
        else:
            self.save_grasp_image_matplotlib(show_pc, grasp, vis_save_path, self.detect_num) # 畫最高分grasp
        vis_time = time.time()
        print("vis time = ", vis_time - infer_time)
        return final_grasp, grasp[:3], show_pc

    def com_z_score(self, grasp):
        grasp_z = np.matmul(grasp[:3,:3], np.array([1,0,0]).reshape(-1,1)).reshape(-1)
        z = np.array([0,0,-1])
        # score_angle = np.abs(np.dot(grasp_z, z))
        # print(score_angle)
        sin_angle = np.dot(grasp_z, z)
        score_angle = 0.5+np.asin(sin_angle)/np.pi
        return score_angle

    def com_z_score_torch(self, grasp):
        # grasp_y = torch.bmm(grasp[:,:3,:3].float(), torch.tensor([[0,1,0]]).float().repeat(len(grasp),1).view(-1,3,1)).transpose(2,1)
        # y = grasp_y.clone().view(len(grasp),3,1)
        # y[:,2,0] = 0

        # grasp_x = torch.bmm(grasp[:,:3,:3].float(), torch.tensor([[1,0,0]]).float().repeat(len(grasp),1).view(-1,3,1)).transpose(2,1)
        # z = torch.tensor([[0,0,-1]]).repeat(len(grasp),1).view(-1,3,1).float()
        # score_angle = (torch.abs(torch.bmm(grasp_y, y)).view(-1) + torch.abs(torch.bmm(grasp_x, z)).view(-1))/2
        # return score_angle

        if grasp.is_cuda:
            grasp = grasp.cpu()
        grasp_z = torch.bmm(grasp[:,:3,:3].float(), torch.tensor([[1,0,0]]).float().repeat(len(grasp),1).view(-1,3,1)).transpose(2,1)
        z = torch.tensor([[0,0,-1]]).repeat(len(grasp),1).view(-1,3,1).float()
        # score_angle = torch.abs(torch.bmm(grasp_z, z)).view(-1)
        sin_angle = torch.bmm(grasp_z, z).view(-1)
        score_angle = 0.5+torch.asin(sin_angle)/np.pi
        return score_angle

    def inv_transform_grasp(self, grasp_trans):
        '''
        Input:
            grasp_trans:[B, 9] (center[3], axis_y[3], grasp_angle[1], \
                                        antipodal score[1], center score[1])
        Output:
            matrix: [B, 3, 4] 
                    [[x1, y1, z1, c1],
                    [x2, y2, z2, c2],
                    [x3, y3, z3, c3]]
            grasp_score_ori: [B]
        '''
        grasp_trans = torch.Tensor(grasp_trans).float()
        no_grasp_mask = (grasp_trans.view(-1,9)[:,-1] == -1)
        # no_grasp_mask = (grasp_trans.view(-1,9)[:,-1] == -2)

        center = grasp_trans.view(-1,9)[:,:3]
        axis_y = grasp_trans.view(-1,9)[:,3:6]
        angle = grasp_trans.view(-1,9)[:,6]
        cos_t, sin_t = torch.cos(angle), torch.sin(angle)

        B = len(grasp_trans.view(-1,9))
        R1 = torch.zeros((B, 3, 3))
        for i in range(B):
            r = torch.tensor([[cos_t[i], 0, -sin_t[i]],[0, 1, 0],[sin_t[i], 0, cos_t[i]]]).view(1,3,3)
            R1[i,:,:] = r

        norm_y = torch.norm(axis_y, dim=1)
        axis_y = torch.div(axis_y, norm_y.view(-1,1))
        zero = torch.zeros((B, 1), dtype=torch.float32)
        if axis_y.is_cuda:
            axis_y[torch.nonzero(torch.eq(norm_y, 0))] = torch.tensor(([0,1,0]), dtype=torch.float).cuda()
            zero = zero.cuda()
            R1 = R1.cuda()
        else:
            axis_y[torch.nonzero(torch.eq(norm_y, 0))] = torch.tensor(([0,1,0]), dtype=torch.float)
        axis_x = torch.cat((axis_y[:, 1].view(-1,1), -axis_y[:, 0].view(-1,1), zero), 1)
        norm_x = torch.norm(axis_x, dim=1)
        axis_x = torch.div(axis_x, norm_x.view(-1,1))
        if axis_y.is_cuda:
            axis_x[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float).cuda()
        else:
            axis_x[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float)

        axis_z = torch.cross(axis_x, axis_y, dim=1)
        norm_z = torch.norm(axis_z, dim=1)
        axis_z = torch.div(axis_z, norm_z.view(-1,1))
        if axis_z.is_cuda:
            axis_z[torch.nonzero(torch.eq(norm_z, 0))] = torch.tensor(([0,0,1]), dtype=torch.float).cuda()
        else:
            axis_z[torch.nonzero(torch.eq(norm_z, 0))] = torch.tensor(([0,0,1]), dtype=torch.float)
        matrix = torch.cat((axis_x.view(-1,3,1), axis_y.view(-1,3,1), axis_z.view(-1,3,1)), dim=2)
        matrix = torch.bmm(matrix, R1)
        approach = matrix[:,:,0]
        norm_x = torch.norm(approach, dim=1)
        approach = torch.div(approach, norm_x.view(-1,1))
        if approach.is_cuda:
            approach[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float).cuda()
        else:
            approach[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float)

        minor_normal = torch.cross(approach, axis_y, dim=1)
        # center = center - 0.02*approach


        # point[2] + frame[2, 0] * self.depth
        matrix = torch.cat((approach.view(-1,3,1), axis_y.view(-1,3,1), minor_normal.view(-1,3,1), center.view(-1,3,1)), dim=2)#.permute(0,2,1)
        matrix[no_grasp_mask] = -1
        matrix = matrix.view(len(grasp_trans), 3, 4)
        
        grasp_score_ori = grasp_trans[:,7]
        grasp_score_ori = grasp_score_ori.view(-1)
        grasp_score_ori[no_grasp_mask] = -1
        grasp_score_ori = grasp_score_ori.view(len(grasp_trans), -1)


        grasp_center_ori = grasp_trans[:,8]
        grasp_center_ori = grasp_center_ori.view(-1)
        grasp_center_ori[no_grasp_mask] = -1
        grasp_center_ori = grasp_center_ori.view(len(grasp_trans), -1)

        keep_indx = torch.arange(len(matrix))
        keep_indx = (matrix[:,0,3] < self.bounds[1]-0.1) & (matrix[:,0,3] > self.bounds[0]+0.1) & \
                       (matrix[:,1,3] < self.bounds[3]-0.1) & (matrix[:,1,3] > self.bounds[2]+0.1)
        return matrix[keep_indx], grasp_score_ori.view(-1)[keep_indx].numpy(), grasp_center_ori.view(-1)[keep_indx].numpy()

    def _local_to_global_transformation_mat(self):
        T_local_to_global = np.eye(4)
        if args.camera == 'kinect2':
            euler_deg = args.camera_transform['euler_deg']
            translation = args.camera_transform['translation']
            euler_rad = np.radians(euler_deg)
            rot_mat = transforms3d.euler.euler2mat(*euler_rad)
            T_local_to_global = np.eye(4)
            T_local_to_global[0:3, 0:3] = rot_mat
            T_local_to_global[0:3, 3] = np.array(translation)
        elif args.camera == 'our':
            euler_deg = args.camera_transform['euler_deg']
            translation = args.camera_transform['translation']
            euler_rad = np.radians(euler_deg)
            rot_mat = transforms3d.euler.euler2mat(*euler_rad)
            T_local_to_global = np.eye(4)
            T_local_to_global[0:3, 0:3] = rot_mat
            T_local_to_global[0:3, 3] = np.array(translation)



        elif args.camera == 'kinectdk_right':
            T_local_to_global = np.array([[ 0.85821133,  0.14616676, -0.49204532,  1.1],
                                    [ 0.5132859,  -0.23821283,  0.82449514, -0.5],
                                    [ 0.00330227, -0.96015099, -0.27946228,  1.55],
                                    [ 0, 0, 0,  1]])

        elif args.camera == 'kinectdk_left':
            T_local_to_global = np.array([[-0.86182764,  0.35765956, -0.35962862,  1],
                                    [ 0.50080489,  0.48781346, -0.71500524,  0.8],
                                    [-0.08029678, -0.79631505, -0.59952878,  2.1],
                                    [ 0, 0, 0,  1]])

        # T_local_to_global = np.linalg.inv(T_local_to_global)
        # print(T_local_to_global)
        return T_local_to_global  

    def _get_show_pc(self, pc, real_data):
        #@ pc: [N, 6]
        pc_xyz, pc_color = pc[:,:3], pc[:,3:6]
        if real_data:
            pc_xyz = np.matmul(self._local_to_global_transformation_mat(), \
                            np.c_[pc_xyz, np.ones([len(pc_xyz), 1])].T).T[:,:3]                
        pc = np.c_[pc_xyz, pc_color]
        if real_data:
            pc = pc[(pc[:,0] < self.bounds[1]) & (pc[:,0] > self.bounds[0])]
            pc = pc[(pc[:,1] < self.bounds[3]) & (pc[:,1] > self.bounds[2])]
            pc = pc[pc[:,2] > self.table_height-0.1]
            if 'kinectdk' in args.camera:
                pc = pc[pc[:,2] < self.table_height + 0.7]

            
        if len(pc) >= self.all_points_num:
            select_point_index = np.random.choice(len(pc), 10000, replace=False)
        else:
            select_point_index = np.random.choice(len(pc), 10000, replace=True)
        pc = pc[select_point_index]
        return pc

    def _pretreat_pc(self, pc, real_data):
        def _noise_color(color):
            obj_color_time = 0.9 - np.random.rand(3) * 0.15
            print("noise color time", obj_color_time)
            for i in range(3):
                color[:,i] *= obj_color_time[i]
            return color

        #@ pc: [N, 6] 
        pc_xyz, pc_color = pc[:,:3], pc[:,3:6]
        if real_data:
            pc_xyz = np.matmul(self._local_to_global_transformation_mat(), \
                            np.c_[pc_xyz, np.ones([len(pc_xyz), 1])].T).T[:,:3]  
        pc = np.c_[pc_xyz, pc_color]
        print(np.mean(pc, axis=0) )
        print(pc[:,0].max(), pc[:,0].min())
        print(pc[:,1].max(), pc[:,1].min())
        print(pc[:,2].max(), pc[:,2].min())

        if real_data:
            print(self.bounds)
            pc = pc[(pc[:,0] < self.bounds[1]) & (pc[:,0] > self.bounds[0])]
            pc = pc[(pc[:,1] < self.bounds[3]) & (pc[:,1] > self.bounds[2])]
            pc = pc[pc[:,2] > self.table_height-0.1]
            if 'kinectdk' in args.camera:
                pc = pc[pc[:,2] < self.table_height + 0.7]
        if args.camera == 'kinect2':
            pc[:,2] = (pc[:,2]-np.mean(pc[:,2])-0.01)*1.2 + np.mean(pc[:,2])+0.01
        if args.camera == 'our':
            # 把z軸差距拉大，不使用的話效果差滿多的，算是一種前處理?增加Z軸差異用
            pc[:,2] = (pc[:,2]-np.mean(pc[:,2])-0.01)*args.z_rescale + np.mean(pc[:,2])+0.01
        print(np.mean(pc, axis=0) )

        select_point_index = None
        if len(pc) >= self.all_points_num:
            select_point_index = np.random.choice(len(pc), self.all_points_num, replace=False)
        elif len(pc) < self.all_points_num:
            select_point_index = np.random.choice(len(pc), self.all_points_num, replace=True)
        pc = pc[select_point_index]
        show_pc = copy.deepcopy(pc)
        # pc[:,3:6] = _noise_color(pc[:,3:6])
        return pc, show_pc


def main():
    import glob
    # pc_paths = glob.glob("./test_file/*/*1.npy")
    pc_paths = glob.glob(args.pc_paths)
    destination_folder = args.vis_output + args.exp_name + '/'
    os.makedirs(destination_folder, exist_ok=True)
    
    # pc_paths = glob.glob("test_file/*/*2.npy")
    # args.camera = 'kinectdk_right'

    # pc_paths = glob.glob("test_file/*/*3.npy")
    # args.camera = 'kinectdk_left'
    GDetector = GraspDetector(args.method)

    for pc_path in pc_paths:
        start_time = time.time()
        pc = np.zeros((20,6))
        if '.npy' in pc_path:
            pc = np.load(pc_path)
        elif '.p' in pc_path:
            pc = np.load(pc_path, allow_pickle=True)['points']
        GDetector.generate_grasp_from_file(pc, pc_path)
        end_time = time.time()
        print("一次時間 = ", end_time - start_time)
    # pc = np.load("/home/tomyeh/env/REGNet-v2/REGNet-V2/test_file/our_1.npy")
    # GDetector.generate_grasp_from_file(pc, "/home/tomyeh/env/REGNet-v2/REGNet-V2/test_file/our_1.npy")
    shutil.copy("config.yaml", destination_folder)

if __name__ == "__main__":
    main()
