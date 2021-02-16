# Credit: https://github.com/naoto0804/pytorch-AdaIN/blob/master/function.py
import torch.nn as nn


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def attribute_summary_instance_normalization(content_feat, style_mean, style_std):
    size = content_feat.size()
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


# This is not a correct implementation, keep for reference only.
class ASIN(nn.Module):
    def __init__(self, style_mean, style_std):
        super(ASIN, self).__init__()
        self.style_mean = style_mean
        self.style_std = style_std

    def forward(self, content_feat):
        size = content_feat.size()
        content_mean, content_std = calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * self.style_std.expand(size) + self.style_mean.expand(size)
