from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from utils import send_message


def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out


def get_means_stds(cluster_npz_path, batch_size, device):
    clusters = np.load(cluster_npz_path)
    means = clusters["centers"]
    stds = clusters["stds"]
    means = [means for _ in range(batch_size)]
    stds = [stds for _ in range(batch_size)]
    means = np.stack(means, axis=1)
    stds = np.stack(stds, axis=1)
    means = torch.tensor(means).to(device)
    stds = torch.tensor(stds).to(device)
    return means, stds


class Solver(object):
    """Solver for training and testing StarGAN.
    For training, we use the Adam optimizer, a mini-batch size
    of 32, a learning rate of 0.0001, and decay rates of β1 = 0:5,
    β2 = 0:999.

    As a module to predict the affine parameters for ASIN, our multilayer
    perceptron consists of seven layers for FFHQ and
    EmotioNet datasets and three layers for the CelebA dataset.
    """

    def __init__(self, celeba_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.celeba_loader = celeba_loader

        # Model configurations.
        # self.c_dim = config.c_dim  # the domain number
        self.c_dim = len(config.selected_clusters_train)
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.lambda_lnt = config.lambda_lnt

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters
        self.test_images_num = config.test_images_num

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = os.path.join(config.exp_dir, config.exp_id, "logs")
        self.sample_dir = os.path.join(config.exp_dir, config.exp_id, "samples")
        self.model_save_dir = os.path.join(config.exp_dir, config.exp_id, "models")
        self.result_dir = os.path.join(config.exp_dir, config.exp_id, "results")

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        self.means, self.stds = get_means_stds(config.cluster_npz_path, self.batch_size, self.device)

        self.selected_clusters_test = config.selected_clusters_test
        if len(self.selected_clusters_test) == 0:
            self.selected_clusters_test = [i for i in range(self.means.shape[0])]

        # Classification loss
        self.pos_weight = torch.ones([self.c_dim], device=self.device) * self.c_dim

        # Create a generator and a discriminator.
        # TODO: fully control the generator's parameters
        self.G = Generator(self.g_conv_dim, self.g_repeat_num)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, (self.beta1, self.beta2))
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, (self.beta1, self.beta2))
        print_network(self.G, 'G')
        print_network(self.D, 'D')

        self.G.to(self.device)
        self.D.to(self.device)

        # Use tensorboard.
        if self.use_tensorboard:
            from logger import Logger
            self.logger = Logger(self.log_dir)

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters), end=' ')
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        print("Done.")

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""
        if self.dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False,
                                                      pos_weight=self.pos_weight) / logit.size(0)
        elif self.dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        data_loader = self.celeba_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, _, _, _, _ = next(data_iter)
        x_fixed = x_fixed.to(self.device)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            # TODO: is it okay to iterate data like that?
            try:
                x_real, _, label, mean, std = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                x_real, _, label, mean, std = next(data_iter)

            x_real = x_real.to(self.device)  # Input images.
            label = label2onehot(label, self.c_dim)
            label = label.to(self.device)
            mean = mean.to(self.device)
            std = std.to(self.device)

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            # (batch_size, 1, image_size/conv_dim, image_size/conv_dim), (batch_size, c_dim)
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label)

            # Compute loss with fake images.
            x_fake, _ = self.G(x_real, mean, std)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {'D/loss': d_loss.item(),
                    'D/loss_real': d_loss_real.item(),
                    'D/loss_fake': d_loss_fake.item(),
                    'D/loss_cls': d_loss_cls.item(),
                    'D/loss_gp': d_loss_gp.item()}

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i + 1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake, h_x_real = self.G(x_real, mean, std)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label)

                # Target-to-original domain.
                x_reconst, h_x_fake = self.G(x_fake, mean, std)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Latent loss: the distance between real and fake images in the feature space
                g_loss_lnt = torch.mean(torch.sqrt(torch.square(h_x_real - h_x_fake)))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls + self.lambda_lnt * g_loss_lnt
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss'] = g_loss.item()
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                loss['G/loss_lnt'] = g_loss_lnt.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i + 1)
            # Translate fixed images for debugging.
            if (i + 1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for select_cluster in self.selected_clusters_test:
                        mean, std = self.means[select_cluster], self.stds[select_cluster]
                        x_fake, _ = self.G(x_fixed, mean, std)
                        x_fake_list.append(x_fake)
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i + 1))
                    save_image(denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i + 1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))
                send_message(f'Current iteration step is {i + 1}.')

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        for test_iter in self.test_iters:
            # Prepare target dir
            if len(self.selected_clusters_test) == self.means.shape[0]:
                label_str = 'all'
            else:
                label_str = ' '.join(str(x) for x in self.selected_clusters_test)
            target_dir = os.path.join(self.result_dir, f"model-{test_iter}", label_str)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            # Load the trained generator.
            self.restore_model(test_iter)
            # Set data loader.
            data_loader = self.celeba_loader

            with torch.no_grad():
                for i, (x_real, _, _, _, _) in enumerate(data_loader):

                    # Prepare input images and target domain labels.
                    x_real = x_real.to(self.device)

                    # Translate images.
                    x_fake_list = [x_real]

                    for select_cluster in self.selected_clusters_test:
                        mean, std = self.means[select_cluster], self.stds[select_cluster]
                        x_fake, _ = self.G(x_real, mean, std)
                        x_fake_list.append(x_fake)

                    # Save the translated images.
                    x_concat = torch.cat(x_fake_list, dim=3)
                    result_path = os.path.join(target_dir, f'image{i + 1}.jpg')
                    save_image(denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(result_path))
                    if i == self.test_images_num:
                        break
