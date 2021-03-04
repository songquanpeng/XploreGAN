#!/bin/bash

RESUME_EXP_ID=$1
RESUME_ITER=$2

CUDA_VISIBLE_DEVICES=1 python main.py \
--mode train --dataset CelebA --image_size 128 \
--cluster_npz_path data/celeba/generated/clusters.npz \
--dataset_path ../celeba/images \
--batch_size 32 \
--selected_clusters_train 84 3 44 7 8 59 10 18 29 91 \
--selected_clusters_test 84 3 44 7 8 59 10 18 29 91 \
--resume_iters $RESUME_ITER \
--exp_id $RESUME_EXP_ID