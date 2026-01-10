import os

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def loaddata(data_dir):
    """Load and transform image datasets for training, validation, and testing.

    Args:
        data_dir (str): Path to the root data directory containing
            'train', 'valid', and 'test' subdirectories.

    Returns:
        tuple: A tuple containing:
            - dataloaders (dict): DataLoaders for 'train', 'val', and 'test'.
            - image_datasets (dict): ImageFolder datasets for each split.

    Raises:
        FileNotFoundError: If the train directory does not exist.
        FileNotFoundError: If the valid directory does not exist.
        FileNotFoundError: If the test directory does not exist.
    """
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f'The directory {train_dir} does not exist')
    if not os.path.exists(valid_dir):
        raise FileNotFoundError(f'The directory {valid_dir} does not exist')
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f'The directory {test_dir} does not exist')

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'val': datasets.ImageFolder(valid_dir, data_transforms['val']),
        'test': datasets.ImageFolder(test_dir, data_transforms['test'])
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'val': DataLoader(image_datasets['val'], batch_size=64, shuffle=True),
        'test': DataLoader(image_datasets['test'], batch_size=64, shuffle=True)
    }

    return dataloaders, image_datasets


def imshow(image, ax=None, title=None, normalize=True):
    """Display a PyTorch tensor as an image.

    Args:
        image: PyTorch tensor of shape (C, H, W) representing an image.
        ax: Matplotlib axes object to plot on. If None, creates a new figure.
        title: Title for the image (currently unused).
        normalize: If True, denormalize the image using ImageNet mean and std.

    Returns:
        The matplotlib axes object with the displayed image.
    """
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


def format_time(seconds):
    """Format time in seconds to h:m:s string.

    Args:
        seconds (float): Time in seconds.

    Returns:
        str: Formatted time string in format "Xh Ym Zs".
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"
