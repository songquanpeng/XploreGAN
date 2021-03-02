#!/bin/bash

RESUME_EXP_ID=$1
RESUME_ITER=$2

CUDA_VISIBLE_DEVICES=0 python main.py \
--mode train --dataset CelebA --image_size 128 \
--cluster_npz_path data/celeba/generated/clusters.npz \
--dataset_path ../celeba/images \
--batch_size 32 --selected_clusters_test 4 7 10 11 15 32 \
--resume_iters $RESUME_ITER \
--exp_id $RESUME_EXP_ID