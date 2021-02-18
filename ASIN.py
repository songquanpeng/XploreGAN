# Credit: https://github.com/naoto0804/pytorch-AdaIN/blob/master/function.py

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def calc_mean_std_from_style(style):
    # the shape should be [batch_size, 256, 1, 1]
    size = style.size()
    n, c = size[:2]
    m = c // 2
    mean = style[:, :m].view(n, m, 1, 1)
    std = style[:, m:].view(n, m, 1, 1)
    return mean, std


def attribute_summary_instance_normalization(content_feat, style):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std_from_style(style)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)
