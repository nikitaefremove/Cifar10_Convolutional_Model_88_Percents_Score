import torch.nn as nn


def advanced_conv_cifar():
    # we have 32 * 32 * 3 after resizing
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),  # 32 x 32 x 32
        nn.ReLU(),
        nn.BatchNorm2d(32),

        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),  # 32 x 32 x 32
        nn.ReLU(),
        nn.BatchNorm2d(32),

        nn.MaxPool2d(2),  # 16 x 16 x 32
        nn.Dropout2d(p=0.25),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # 16 x 16 x 64
        nn.ReLU(),
        nn.BatchNorm2d(64),

        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),  # 16 x 16 x 64
        nn.ReLU(),
        nn.BatchNorm2d(64),

        nn.MaxPool2d(2),  # 8 x 8 x 64
        nn.Dropout2d(p=0.25),

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # 8 x 8 x 128
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),  # 8 x 8 x 128
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.MaxPool2d(2),  # 4 x 4 x 128
        nn.Dropout2d(p=0.25),

        nn.Flatten(),

        nn.Linear(4 * 4 * 128, 512),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.25),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
