import torch
import torch.nn.functional as F


def calc_mean_std(features):
    """

    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """

    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-8
    return features_mean, features_std


def adain(content_features, style_features):
    """
    Adaptive Instance Normalization
    This function will be used in the hierarchical AutoEncoder
    to colorize the reconstructed decoder features using encoder style features

    :param content_features: shape -> [batch_size, c, h, w]
    :param style_features: shape -> [batch_size, c, h, w]
    :return: normalized_features shape -> [batch_size, c, h, w]
    """
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    features = style_std * (content_features - content_mean) / content_std + style_mean
    return features


def style_decorator(content_features, style_features, patch_size=5, alpha=1):
    """
    A style decorator function only using zca normalization/colorization

    :param content_features:
    :param style_features:
    :param patch_size:
    :param alpha:
    :return:

    """
    # features projection by normalization
    projected_content_features, _, _ = zca_normalization(content_features)
    projected_style_features, style_kernel, mean_style_features = zca_normalization(style_features)

    # matching and reassembling
    rearranged_features = style_swap(projected_content_features, projected_style_features, patch_size)

    # feature reconstruction by colorization
    # rearranged_features = projected_content_features
    reconstruction_features = zca_colorization(rearranged_features, style_kernel, mean_style_features)
    reconstruction_features = alpha * reconstruction_features + (1 - alpha) * projected_content_features

    return reconstruction_features


def zca_normalization(features):
    """
    A zca_normlaization function to normalize features
    NOTICE that colorization_kernel and features_mean are only used to colorize
    the normalized content features(maybe after style swap) with style features

    :param features: 1 * C * H * W
    :return: normalized_features: (1 * C * H * W)
              colorization_kernel:  (C * C)
              features_mean: (C * 1)
    """
    features = features.squeeze(0)
    c, h, w = features.shape
    features = features.reshape(c, -1)
    features_mean = torch.mean(features, 1, keepdim=True)
    unbiased_features = features - features_mean
    gram = unbiased_features @ unbiased_features.t() #/ (h * w - 1)
    u, e, v = torch.svd(gram)

    # if necessary, use k-th largest eig-value
    k_c = c
    for i in range(c):
        if e[i] < 0.00001:
            k_c = i
            break

    d_inv = e[:k_c].pow(-0.5)
    d = e[:k_c].pow(0.5)

    colorization_kernel = v[:, :k_c] @ torch.diag(d) @ v[:, :k_c].t()

    normalized_features = v[:, :k_c] @ torch.diag(d_inv) @ v[:, :k_c].t() @ unbiased_features
    normalized_features = normalized_features.reshape(1, c, h, w)
    return normalized_features, colorization_kernel, features_mean


def zca_colorization(normalized_features, colorization_kernel, mean_feaures):
    """
    A colorization function to colorize content features

    :param normalized_features: (1 * C * H * W)
    :param colorization_kernel: (C * C)
    :param mean_feaures: (C * 1)

    :return: colorized_features: (1 * C * H * W)
    """
    features = normalized_features.squeeze(0)
    c, h, w = features.shape
    features = features.reshape(c, -1)
    colorized_features = colorization_kernel @ features + mean_feaures
    colorized_features = colorized_features.reshape(1, c, h, w)
    return colorized_features


def style_swap(content_feature, style_feature, kernel_size=5, stride=1):
    """
    Nearest patch swapping function after features normalization

    Kernel_size here is equivalent to extracted patch size

    :param content_feature: (1, C, H, W)
    :param style_feature:  (1, C, H, W)
    :param kernel_size: 3 or 5 recommended
    :param stride: default as 1

    :return: res: (1, C, H, W)
    """
    # extract patches from style_feature with shape (1, C, H, W)
    kh, kw = kernel_size, kernel_size
    sh, sw = stride, stride
    patches = style_feature.unfold(2, kh, sh).unfold(3, kw, sw)
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    patches = patches.reshape(-1, *patches.shape[-3:])  # (patch_numbers, C, kh, kw)

    # calculate Frobenius norm and normalize the patches at each filter
    norm = torch.norm(patches.reshape(patches.shape[0], -1), dim=1).reshape(-1, 1, 1, 1)
    noramalized_patches = patches / norm

    conv_out = F.conv2d(content_feature, noramalized_patches)

    # calculate the argmax at each spatial location, which means at each (kh, kw),
    # there should exist a filter which provides the biggest value of the output
    one_hots = torch.zeros_like(conv_out)
    one_hots.scatter_(1, conv_out.argmax(dim=1, keepdim=True), 1)

    # deconv/transpose conv
    deconv_out = F.conv_transpose2d(one_hots, patches)

    # calculate the overlap from deconv/transpose conv
    overlap = F.conv_transpose2d(one_hots, torch.ones_like(patches))

    # average the deconv result
    res = deconv_out / overlap
    return res


# c = torch.arange(27).reshape(1, 3, 3, 3).float()
# s = c + 1e-4
# res = style_decorator(c, s, 3, 1)
