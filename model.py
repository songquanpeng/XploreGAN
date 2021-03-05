import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ASIN import attribute_summary_instance_normalization as ASIN
from layers import NoiseLayer


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """
    Adapted from StarGAN [4], our encoder
    has two convolutional layers for downsampling followed
    by six residual blocks [14] with spectral normalization
    [28]. Our decoder has six residual blocks with attribute
    summary instance normalization (ASIN), with per pixel
    noise [21] added after each convolutional layer. It is
    followed by two transposed convolutional layers for upsampling.
    We also adopt stochastic variation [21] to increase
    generation performance on fine, stochastic details of the image.
    """

    def __init__(self, conv_dim=64, repeat_num=6, pca_dim=256, mlp_layer_num=3, mlp_neurons_num=256):
        super(Generator, self).__init__()

        # MLP part.
        # Used to predict the affine parameters for ASIN (7 layers for FFHQ & EmotionNet, 3 layer for CelebA)
        layers = []
        # TODO: choose a suitable mlp_neurons_num
        layers.append(nn.Linear(2 * pca_dim, mlp_neurons_num))
        layers.append(nn.ReLU())
        for i in range(mlp_layer_num - 2):
            layers.append(nn.Linear(mlp_neurons_num, mlp_neurons_num))
            layers.append(nn.ReLU())
        # Here we hardcode the output dimension
        layers.append(nn.Linear(mlp_neurons_num, 2 * 256))
        # TODO: does we need a activation layer here? if yes, which one?
        # layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

        # Encoder part.
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            # TODO: use spectral normalization here
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.encoder = nn.Sequential(*layers)

        # Decoder part 1.
        self.decoder_residual_blocks = nn.ModuleList()
        # TODO: Six residual blocks with attribute summary instance normalization (ASIN).
        for i in range(repeat_num):
            self.decoder_residual_blocks.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Decoder part 2.
        layers = []
        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            # layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
            # Add per pixel noise after each convolutional layer.
            layers.append(NoiseLayer(curr_dim // 2))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.decoder_part2 = nn.Sequential(*layers)

    def forward(self, x, mean, std):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        # TODO: the shape of mean and std should be what?
        meta = torch.cat((mean, std), 1).float()
        style = self.mlp(meta)
        x = self.encoder(x)
        h = x.detach().clone()
        # decoder part 1
        # TODO: not sure if it's okay
        for residual_block in self.decoder_residual_blocks:
            x = residual_block(x)
            x = ASIN(x, style)
        x = self.decoder_part2(x)
        return x, h


class Discriminator(nn.Module):
    """
    For the discriminator, we use PatchGANs [24, 18, 43]
    to classify whether image patches are real or fake.
    """

    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
