import torch


# Normalization function. Return mean and standard deviation
def get_normalize(features: torch.Tensor):
    # size [N, C, H, W]
    # N is amount of objects, C — amount of chanel, H, W — size of image
    # return mean by chanel and standard deviation by chanel

    means = features.mean(dim=(0, 2, 3))
    stds = features.std(dim=(0, 2, 3))

    return means, stds
