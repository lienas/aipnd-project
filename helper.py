import os

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torchvision.models as models
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device(use_gpu=True):
    """Determine the best available device.

    Args:
        use_gpu (bool): If True, attempt to use GPU (MPS or CUDA).
            If False, always use CPU.

    Returns:
        torch.device: The selected device (mps, cuda, or cpu).
    """
    if use_gpu:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("Warning: GPU requested but not available. Using CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    return device


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


def format_time(seconds, show_ms=False):
    """Format time in seconds to h:m:s string.

    Args:
        seconds (float): Time in seconds.
        show_ms (bool): If True, include milliseconds in output.

    Returns:
        str: Formatted time string in format "Xh Ym Zs" or "Xh Ym Zs Wms".
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if show_ms:
        ms = int((seconds % 1) * 1000)
        return f"{hours}h {minutes}m {secs}s {ms}ms"
    return f"{hours}h {minutes}m {secs}s"


def load_checkpoint(filepath, use_gpu=True):
    """Load a model checkpoint and rebuild the model.

    Args:
        filepath (str): Path to the checkpoint file (.pth).
        use_gpu (bool): If True, use GPU if available. If False, use CPU.

    Returns:
        tuple: (model, device) - The reconstructed model with loaded weights,
            classifier, and class_to_idx mapping, plus the device it's on.
    """
    device = get_device(use_gpu)
    checkpoint = torch.load(filepath, weights_only=False, map_location=device)
    model = getattr(models, checkpoint['arch'])(weights=None)

    # Dynamically set classifier (works for VGG/AlexNet and ResNet)
    classifier_attr = checkpoint.get('classifier_attr', 'classifier')
    setattr(model, classifier_attr, checkpoint['classifier'])
    model.classifier_attr = classifier_attr  # Store for later access

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.to(device)
    return model, device


def process_image(image_path):
    """Process an image for use with a PyTorch model.

    Scales the image so the shortest side is 256 pixels, center crops
    to 224x224, and normalizes using ImageNet mean and std.

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: Processed image tensor of shape (3, 224, 224),
            normalized with ImageNet statistics.
    """
    pil_img = Image.open(image_path)

    width, height = pil_img.size
    if width < height:
        new_width = 256
        new_height = int(height * 256 / width)
    else:
        new_height = 256
        new_width = int(width * 256 / height)
    pil_img = pil_img.resize((new_width, new_height))

    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = left + 224
    bottom = top + 224
    pil_img = pil_img.crop((left, top, right, bottom))

    np_image = np.array(pil_img) / 255.0

    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    np_image = (np_image - mean) / std

    np_image = np_image.transpose((2, 0, 1))

    return torch.from_numpy(np_image).float()
