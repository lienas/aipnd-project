import matplotlib.pyplot as plt
import numpy as np


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
