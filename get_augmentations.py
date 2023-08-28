import torchvision.transforms as T


# Function for augmentation.
# Resize image,  make augmentation for train (crop, b&w, blur, random)
# Make tensor from train and test and normalize
def get_augmentations(train: bool = True) -> T.Compose:
    #     means = (dataset_train.data / 255).mean(axis=(0, 1, 2))
    #     stds = (dataset_train.data / 255).std(axis=(0, 1, 2))
    means = (0.49139968, 0.48215841, 0.44653091)
    stds = (0.24703223, 0.24348513, 0.26158784)

    if train:
        return T.Compose([
            T.RandAugment(),
            T.ToTensor(),
            T.Normalize(mean=means, std=stds)
        ]
        )
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=means, std=stds)
        ]
        )
