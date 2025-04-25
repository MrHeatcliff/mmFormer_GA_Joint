#!/bin/bash

# Tên dataset và nơi lưu kết quả
dataname='BRATS2018'
savepath='output'
datapath='/tran_tien_dat_112/workspace/sanglequang/brats2018/versions/10/BRATS2018_Training_none_npy'

PYTHON=python3

export CUDA_VISIBLE_DEVICES=0

# Cấu hình CUDA 11.8 (nếu cần thủ công)
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Chạy training
$PYTHON train.py \
  --batch_size=1 \
  --datapath $datapath \
  --savepath $savepath \
  --num_epochs 1000 \
  --dataname $dataname

# Nếu muốn chạy eval sau khi train xong:
# resume=output/model_last.pth
# $PYTHON train.py --batch_size=1 --datapath $datapath --savepath $savepath --num_epochs 0 --dataname $dataname --resume $resume
