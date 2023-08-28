import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.inference_mode()
def predict_tta(model: nn.Module,
                loader: DataLoader,
                device: torch.device,
                iterations: int = 2):
    predictions = []
    model = model.to(device)
    model.eval()  # set the model to evaluation mode

    for i in range(iterations):
        single_prediction = []
        for x, _ in loader:
            x = x.to(device)
            output = model(x)  # forward pass
            single_prediction.append(output)

        predictions.append(torch.vstack(single_prediction))

    conc = torch.stack(predictions)

    mean = conc.mean(0)

    mean_preds = mean.argmax(dim=1)

    return mean_preds
