# CIFAR10 CNN Image Classification

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify images from the CIFAR10 dataset.

## Framework

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib

## Dataset

CIFAR10 dataset

Classes:

plane
car
bird
cat
deer
dog
frog
horse
ship
truck

## Model

Conv → ReLU → Pool
Conv → ReLU → Pool
FC → FC → FC

## Loss

CrossEntropyLoss

## Optimizer

SGD (lr=0.001, momentum=0.9)

## Run

```bash
pip install torch torchvision matplotlib numpy
python main.py
