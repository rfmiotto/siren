import torch
import torchvision


def get_transforms(size: int = 200):
    transforms = torchvision.transforms.Compose(
        [
            # torchvision.transforms.Resize(size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])),
        ]
    )
    return transforms
