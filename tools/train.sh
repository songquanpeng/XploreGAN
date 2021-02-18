#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--mode train --dataset CelebA --image_size 128 --c_dim 100 \
--cluster_npz_path data/celeba/generated/clusters.npz \
--batch_size 32 --selected_labels 4 7 10 11 15 32