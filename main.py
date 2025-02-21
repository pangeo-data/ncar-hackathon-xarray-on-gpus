#!/usr/bin/env python3
# Simple example to demonstrate how to do training with PyTorch using the ERA5TimeSeriesDataset class.

import os
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import argparse

from ERA5TimeSeriesDataset import ERA5TimeSeriesDataset, PyTorchERA5Dataset

def main():
    num_epochs_default = 2
    batch_size_default = 2
    learning_rate_default = 0.001  # Adjusted for Adam optimizer

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of training epochs.",
        default=num_epochs_default,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Training batch size for one process.",
        default=batch_size_default,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate.",
        default=learning_rate_default,
    )

    argv = parser.parse_args()
    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    learning_rate = argv.learning_rate

    # --------------------------
    # Initialize the ERA5 dataset
    data_path = "/glade/derecho/scratch/ksha/CREDIT_data/ERA5_mlevel_arXiv"
    year_start = 1990
    year_end = 2010
    input_vars = ['t2m', 'V500', 'U500', 'T500', 'Z500', 'Q500']

    era5_dataset = ERA5TimeSeriesDataset(
        data_path=data_path,
        year_start=year_start,
        year_end=year_end,
        input_vars=input_vars,
    )
    
    print(f"Total samples: {len(era5_dataset)}")
    
    # --------------------------
    # Wrap in a PyTorch Dataset
    pytorch_dataset = PyTorchERA5Dataset(era5_dataset)
    data_loader = DataLoader(pytorch_dataset, batch_size=batch_size, shuffle=True)

    print ("Data loaded!")

    # --------------------------
    # Define the U-Net model using segmentation_models_pytorch
    ENCODER = 'resnet50'  # Encoder backbone
    ENCODER_WEIGHTS = 'imagenet'  # Pretrained weights
    CLASSES = input_vars  # Number of output channels (same as input variables)
    ACTIVATION = None  # No activation for regression tasks

    # Create the U-Net model
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=len(input_vars), 
        classes=len(CLASSES),  # Number of output channels
        activation=ACTIVATION,
    )

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    torch.backends.cudnn.benchmark = True

    # --------------------------
    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()  # For regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer
    
    # --------------------------
    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(data_loader):
            # Move tensors to GPU if available
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)  # Shape: (batch_size, num_classes, height, width)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backprop
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], "
                      f"Loss: {running_loss/10:.4f}")
                running_loss = 0.0

    print("Training complete!")

if __name__ == "__main__":
    main()