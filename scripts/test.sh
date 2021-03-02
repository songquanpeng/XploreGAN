#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--mode test --dataset CelebA --image_size 128 --c_dim 100 \
--cluster_npz_path data/celeba/generated/clusters.npz \
--dataset_path ../celeba/images \
--batch_size 32 --selected_clusters_test 4 7 10 11 15 32