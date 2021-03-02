import os
import argparse
from solver import Solver
from dataloader import get_loader
from torch.backends import cudnn
from utils import get_datetime
import json


def main(config):
    # Save config with experiment data.
    target_dir = os.path.join(config.exp_dir, config.exp_id)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    with open(os.path.join(target_dir, "config.json"), 'a') as f:
        print(json.dumps(config.__dict__, sort_keys=True, indent=4), file=f)

    cudnn.benchmark = True

    # Data loader.
    celeba_loader = get_loader(config.cluster_npz_path, config.dataset_path,
                               config.celeba_crop_size, config.image_size, config.batch_size,
                               config.dataset, config.mode, config.num_workers, config.selected_clusters_train)

    # Solver for training and testing StarGAN.
    solver = Solver(celeba_loader, config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    # parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--lambda_lnt', type=float, default=10, help='weight for latent loss')

    # Training configuration.
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Pseudo'])
    parser.add_argument('--selected_clusters_train', nargs='+', type=int, help='selected clusters for training', default=[])

    # Test configuration.
    parser.add_argument('--test_iters', nargs='+', type=int, default=[200000], help='test model from this step')
    parser.add_argument('--selected_clusters_test', nargs='+', type=int, default=[], help='selected labels to generate')
    parser.add_argument('--test_images_num', type=int, default=200, help='set -1 to test all')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=bool, default=True)

    # Directories.
    parser.add_argument('--cluster_npz_path', type=str, default='data/celeba/generated/clusters.npz')
    parser.add_argument('--dataset_path', type=str, default='./data/celeba/')
    parser.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')
    parser.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
    parser.add_argument('--exp_dir', type=str, default='experiments')
    parser.add_argument('--exp_id', type=str, default=get_datetime(), help="experiment id")

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    cfg = parser.parse_args()
    print(cfg)
    main(cfg)
