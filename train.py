"""Trainieren Sie ein neues Netzwerk auf einem Datensatz mit train.py.

Grundlegende Verwendung: python train.py data_directory
Gibt den Trainingsverlust, den Validierungsverlust und
die Validierungsgenauigkeit aus, während das Netzwerk trainiert.

Optionen:
    * Verzeichnis zum Speichern von Checkpoints festlegen:
      python train.py data_dir --save_dir save_directory
    * Architektur wählen: python train.py data_dir --arch "vgg13"
    * Hyperparameter festlegen:
      python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    * GPU für das Training verwenden: python train.py data_dir --gpu
"""

import argparse
import json
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch
import time
import os

import helper

SUPPORTED_MODELS = {
    'vgg16': {
        'loader': models.vgg16,
        'in_features': 25088,
        'classifier': 'classifier'
    },
    'vgg13': {
        'loader': models.vgg13,
        'in_features': 25088,
        'classifier': 'classifier'
    },
    'alexnet': {
        'loader': models.alexnet,
        'in_features': 9216,
        'classifier': 'classifier'
    },
    'resnet18': {
        'loader': models.resnet18,
        'in_features': 512,
        'classifier': 'fc'
    },
    'resnet50': {
        'loader': models.resnet50,
        'in_features': 2048,
        'classifier': 'fc'
    },
}


def train():
    """Train a new network on a dataset."""
    parser = argparse.ArgumentParser(
        description='Train a new network on a dataset'
    )
    parser.add_argument(
        'data_dir',
        type=str,
        help='path to the data directory'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoints',
        help='path to the directory to save the checkpoint'
    )
    parser.add_argument(
        '--arch',
        type=str,
        default='vgg16',
        choices=list(SUPPORTED_MODELS.keys()),
        help='architecture of the model'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='learning rate'
    )
    parser.add_argument(
        '--hidden_units',
        type=int,
        default=1024,
        help='number of hidden units'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='number of epochs'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        default=False,
        help='use GPU for training'
    )
    parser.add_argument(
        '--show_example_images',
        action='store_true',
        default=False,
        help='show example images'
    )

    args = parser.parse_args()

    print(f'train the model using the data in the directory {args.data_dir}')

    # Load the data
    dataloaders, image_datasets = helper.loaddata(args.data_dir)

    # Load the category names
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # Invert mapping: index -> class
    idx_to_class = {v: k for k, v in image_datasets['train'].class_to_idx.items()}

    # Display some images for testing
    if args.show_example_images:
        for i in ['train', 'test', 'val']:
            data_iter = iter(dataloaders[i])
            print(f'data = {i}')
            images, labels = next(data_iter)
            fig, axes = plt.subplots(figsize=(5, 2), ncols=3)
            for ii in range(3):
                ax = axes[ii]
                helper.imshow(images[ii], ax=ax, normalize=True)
                ax.set_title(cat_to_name[idx_to_class[labels[ii].item()]])

        plt.show()

    # Choose the pretrained model and get the number of input features
    model_config = SUPPORTED_MODELS[args.arch]
    model = model_config['loader'](weights='DEFAULT')
    num_features = model_config['in_features']
    classifier_attr = model_config['classifier']

    print(f'Number of input features for the classifier: {num_features}')

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Define the classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(num_features, args.hidden_units)),
        ('relu1', nn.ReLU()),
        ('drop1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(args.hidden_units, 256)),
        ('relu2', nn.ReLU()),
        ('drop2', nn.Dropout(p=0.5)),
        ('fc3', nn.Linear(256, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    # Replace classifier
    setattr(model, classifier_attr, classifier)

    # Use Negative Log-Likelihood Loss
    criterion = nn.NLLLoss()

    # Only train classifier parameters
    classifier_params = getattr(model, classifier_attr).parameters()
    optimizer = optim.Adam(classifier_params, lr=args.learning_rate)

    # train the network

    # Determine device: GPU only if requested and available
    if args.gpu:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("Warning: --gpu flag set but no GPU available. Using CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    model.to(device)

    print(f"Train model on device: {device}")
    print('==============================')

    # set prarameters for training
    epochs = args.epochs
    running_loss = 0
    print_every = 10
    # number of steps on Training_data
    total_steps = len(dataloaders['train'])
    start_time = time.time()

    for epoch in range(epochs):
        steps = 0
        for images, labels in dataloaders['train']:
            steps += 1
            images, labels = images.to(device), labels.to(device)

            logps = model(images)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Validate accuracy every [print_every] steps
            if steps % print_every == 0 or steps == total_steps:
                val_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for images, labels in dataloaders['val']:
                        images, labels = images.to(device), labels.to(device)
                        logps = model(images)

                        loss = criterion(logps, labels)
                        val_loss += loss.item()

                        # calculate the accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(
                    f"Epoch {epoch+1}/{epochs} :: Step {steps}/{total_steps} "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {val_loss/len(dataloaders['val']):.3f}.. "
                    f"Validation accuracy: {accuracy/len(dataloaders['val']):.3f}"
                )

                running_loss = 0
                model.train()
        end_time = time.time()
        elapsed_seconds = end_time - start_time
        print(f"Time elapsed for epoch {epoch+1}: {helper.format_time(elapsed_seconds)}")

    # Save the model
    # Speichern
    checkpoint = {
        'arch': args.arch,  # Modellarchitektur
        'classifier': classifier_attr,
        'state_dict': model.state_dict(),
        'class_to_idx': image_datasets['train'].class_to_idx,
        'optimizer_state': optimizer.state_dict(),
        'epochs': epochs
    }

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Get a name for the checkpoint file
    checkpoint_name = f'checkpoint_{args.arch}_{args.learning_rate}_{args.epochs}.pth'
    checkpoint_path = os.path.join(args.save_dir, checkpoint_name)
    torch.save(checkpoint, checkpoint_path)

    print(f'Checkpoint saved as {checkpoint_path}')


if __name__ == '__main__':
    train()
