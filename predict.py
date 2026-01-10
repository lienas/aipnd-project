#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-project/predict.py
#
# PROGRAMMER: Thomas L.
# DATE CREATED: 01/2026
# REVISED DATE:
# PURPOSE: Predict flower name from an image using a trained deep learning model.
#          Returns the flower name and class probability for a given input image.
#
# Basic usage: python predict.py /path/to/image checkpoint
#
# Options:
#     * Return top K most likely classes:
#       python predict.py input checkpoint --top_k 3
#     * Use a mapping of categories to real names:
#       python predict.py input checkpoint --category_names cat_to_name.json
#     * Use GPU for inference:
#       python predict.py input checkpoint --gpu
##

import argparse
import os
import torch
import time
import matplotlib.pyplot as plt
import json

from helper import process_image, load_checkpoint, imshow, format_time


def predict():

    start_time = time.time()

    parser = argparse.ArgumentParser(description='Predict the class of an image')
    parser.add_argument('input', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top classes to return')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to a JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    parser.add_argument('--imshow', action='store_true', help='Show the image')

    args = parser.parse_args()

    # check if the input image exists
    if not os.path.exists(args.input):
        print(f"The input image {args.input} does not exist.")
        return
    # check if the checkpoint file exists
    if not os.path.exists(args.checkpoint):
        print(f"The checkpoint file {args.checkpoint} does not exist.")
        return
    # check if the category names file exists
    if args.category_names is not None and not os.path.exists(args.category_names):
        print(f"The category names file {args.category_names} does not exist.")
        return
    else:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)

    # load the checkpoint
    model, device = load_checkpoint(args.checkpoint, args.gpu)
    print(f'Using device: {device}')

    # test image preprocessing (optional)
    if args.imshow:
        test_image = process_image(args.input)
        print(test_image.shape)
        imshow(test_image)
        plt.show()

    # Predict the class (or classes) of an image using a trained deep learning model.
    model.eval()

    img = process_image(args.input)
    img = img.unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():
        output = model(img)
        probs = torch.exp(output)  # Convert log-softmax to probabilities

    top_probs, top_indices = probs.topk(args.top_k, dim=1)

    top_probs = top_probs.squeeze().cpu().numpy()
    top_indices = top_indices.squeeze().cpu().numpy()

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]

    print(f'\nPrediction results for: {args.input}')
    print('-' * 50)
    for i, (cls, prob) in enumerate(zip(top_classes, top_probs), start=1):
        print(f'Top{i}: Cat {cls}, {cat_to_name[cls]}, {prob*100:.2f}%')
    print('-' * 50)
    print(f'Inference time: {format_time(time.time() - start_time, show_ms=True)}')


if __name__ == '__main__':
    predict()
