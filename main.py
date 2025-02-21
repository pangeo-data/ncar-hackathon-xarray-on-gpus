#!/usr/bin/env python3
# Simple example to demonstrate how to do training with PyTorch using the ERA5TimeSeriesDataset class.

import os
import time
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
    batch_size_default = 16
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
    mean_path = "/glade/campaign/cisl/aiml/ksha/CREDIT/mean_6h_0.25deg.nc"
    std_path = "/glade/campaign/cisl/aiml/ksha/CREDIT/std_6h_0.25deg.nc"

    #era5_dataset.normalize(mean_path,std_path)

    # --------------------------
    # Wrap in a PyTorch Dataset
    pytorch_dataset = PyTorchERA5Dataset(era5_dataset)
    data_loader = DataLoader(pytorch_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    print ("Data loaded!")

    # --------------------------
    # Define the U-Net model using segmentation_models_pytorch
    ENCODER = 'resnet18'  # Encoder backbone
    ENCODER_WEIGHTS = 'imagenet'  # Pretrained weights
    ENCODER_WEIGHTS = None  # No pretrained weights
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
    #criterion = torch.nn.MSELoss()
    #criterion = torch.nn.L1Loss()
    criterion = nn.SmoothL1Loss(beta=1.0)  # Huber Loss for robust training


    #optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # Adam optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)  

    # --------------------------
    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(data_loader):
            start_time = time.time()  # Start time for the step

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevents exploding gradients

            optimizer.step()
            
            running_loss += loss.item()
            
            step_time = (time.time() - start_time)  # Compute elapsed time in milliseconds
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], "
                  f"Loss: {loss.item():.4f}, Time per step: {step_time:.2f} s.")

    print("Training complete!")

if __name__ == "__main__":
    main()