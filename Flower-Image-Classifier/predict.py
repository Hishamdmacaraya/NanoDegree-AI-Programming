import argparse
import sys
import time
import numpy as np

import torch
from torch import nn, optim
from torchvision import datasets
from torchvision import transforms
from torchvision import models

from PIL import Image
import json

def main():
    global args
    args = parse() 
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled...but no GPU detected")
    if (args.top_k is None):
        top_k = 5
    else:
        top_k = args.top_k
    image_path = args.image_input
    prediction = classify_image(image_path,top_k)
    display_prediction(prediction)
    return prediction

def parse():
    parser = argparse.ArgumentParser(description='Predict flower name from an image along with the probability of that name.')
    parser.add_argument('image_input', type=str, help='Image file path')
    parser.add_argument('checkpoint', type=str, help='Model checkpoint file path')
    parser.add_argument('--top_k', type=int, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Use a mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    return parser.parse_args()

def classify_image(image_path, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Load the model
    checkpoint = torch.load(args.checkpoint)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.eval()
    # Process the image
    image = Image.open(image_path)
    image = process_image(image)
    image = torch.from_numpy(np.array([image])).float()
    # Use GPU if available
    if args.gpu and torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    # Predict the class probabilities
    with torch.no_grad():
        output = model.forward(image)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
        top_p = top_p.cpu().numpy()[0]
        top_class = top_class.cpu().numpy()[0]
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_class = [idx_to_class[i] for i in top_class]
        top_flowers = [cat_to_name[idx_to_class[i]] for i in top_class]
        return list(zip(top_flowers, top_p))

def display_prediction(prediction):
    ''' Display the prediction result.
    '''
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
    else:
        cat_to_name = None
    print('Prediction:')
    for i, (flower, prob) in enumerate(prediction):
        if cat_to_name:
            print('  {}. {}: {:.2f}%'.format(i+1, cat_to_name[flower], prob*100))
        else:
            print('  {}. {}: {:.2f}%'.format(i+1, flower, prob*100))

main()