conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install tensorflow=2.17.0 -y
conda install -c conda-forge open3d transforms3d tqdm tensorboardX opencv -y
pip install fvcore==0.1.3.post20210223

# 解決抓不到nvcc
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
source ~/.bashrc

# python grasp_detect_from_file_multiobjects.py --cuda --gpu 0 --load-model test_assets/multigrasp_layer1/refine_15.model --method multigrasp_layer1 --eval-width 0.06 --use_region True --use_analytic False --camera kinect2