import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10  # https://www.cs.toronto.edu/~kriz/cifar.html
import matplotlib.pyplot as plt
from torch.optim import Adam
from typing import Tuple, List
%matplotlib inline


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    predictions = []
    model = model.to(device)
    model.eval()  # set the model to evaluation mode

    with torch.no_grad():  # deactivate autograd
        for x, _ in loader:  # we do not need 'y' for predictions
            x = x.to(device)
            output = model(x)  # forward pass
            preds = output.argmax(dim=1)  # get the predicted classes
            predictions.append(preds)

    return torch.cat(predictions)