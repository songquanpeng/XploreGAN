#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--mode train --dataset CelebA --image_size 128 --c_dim 100 \
--cluster_npz_path data/celeba/generated/clusters.npz \
--dataset_path ../celeba/images \
--batch_size 32 --selected_labels 4 7 10 11 15 32 \
--selected_clusters 2 4 7 8 10 11 15 16 17 18 19 20 23 24 26 27 28 29 32 40 42 44 46 47 48 51 52 55 56 58 59 67 70 71 72 75 76 78 79 80 81 82 83 85 86 87 88 89 90 91 92 93 96 97 98 99