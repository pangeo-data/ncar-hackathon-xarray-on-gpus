#!/usr/bin/env python3
# Simple example to demonstrate how to do training with PyTorch using the ERA5TimeSeriesDataset class.

import os
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import argparse


from ERA5TimeSeriesDataset import ERA5TimeSeriesDataset, PyTorchERA5Dataset

def create_custom_resnet(input_channels, num_classes=3):
    """
    Creates a ResNet18 model with a configurable number of input channels
    and number of output classes.
    
    - input_channels: int (e.g. len(input_vars))
    - num_classes: int (for the final FC layer output)
    """
    model = models.resnet18(pretrained=False)
    
    # Modify the first conv layer to accept `input_channels` instead of 3
    model.conv1 = nn.Conv2d(
        in_channels=input_channels,
        out_channels=model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False
    )
    
    # Replace the final fully connected layer to match `num_classes`
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def main():
    num_epochs_default = 100
    batch_size_default = 256
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
        "--backend",
        type=str,
        help="Backend for distribted training.",
        default="nccl",
        choices=["nccl", "gloo", "mpi"],
    )

    # --------------------------
    # Initialize the ERA5 dataset
    ## Example usage of the ERA5TimeSeriesDataset class
    data_path = "/glade/derecho/scratch/ksha/CREDIT_data/ERA5_mlevel_arXiv"
    year_start = 1990
    year_end = 2010

    # for now, just surface variables!cd
    input_vars = ['t2m', 'V500', 'U500', 'T500', 'Z500', 'Q500']

    #dataset = ERA5TimeSeriesDataset(data_path, year_start, year_end, input_vars=input_vars)

    era5_dataset = ERA5TimeSeriesDataset(
        data_path=data_path,
        year_start=year_start,
        year_end=year_end,
        input_vars=input_vars,
        #target_vars=target_vars
    )
    
    print(era5_dataset)       
    print(f"Total samples: {len(era5_dataset)}")
    
    # --------------------------
    # Wrap in a PyTorch Dataset
    pytorch_dataset = PyTorchERA5Dataset(era5_dataset)
    data_loader = DataLoader(pytorch_dataset, batch_size=4, shuffle=True, num_workers=2)


    # --------------------------
    # TODO: Normalize the data
    # TODO: Add support for distributed training.

    # --------------------------
    # Create a model (ResNet18) with input channels = number of variables
    n_input_channels = len(input_vars)  # e.g. 3
    # For demonstration, let's suppose we want to output 3 channels
    # In practice, for a full (vars, lat, lon) forecast, you'd need a different model (U-Net, FCN, etc.)
    model = create_custom_resnet(input_channels=n_input_channels, num_classes=3)
    
    # If you have a GPU, move the model to GPU
    print ("CUDA available: ", torch.cuda.is_available())
    print ("CUDA device count: ", torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # --------------------------
    # Set up loss & optimizer
    criterion = nn.MSELoss()  # example: MSE for demonstration
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # --------------------------
    # Training Loop
    epochs = 2  # just a short run for demonstration

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(data_loader):
            # inputs: (batch_size, n_vars, lat, lon)
            # targets: (batch_size, n_vars, lat, lon)
            
            # Move tensors to GPU if available
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)  # shape -> (batch_size, 3) with our custom FC
            # Targets shape -> (batch_size, n_vars, lat, lon)
            
            # Mismatch: 
            # We'll flatten both for a naive demonstration:
            outputs_flat = outputs.view(outputs.size(0), -1)
            targets_flat = targets.view(targets.size(0), -1)[:, :3]  # hack: only compare first 3 values

            loss = criterion(outputs_flat, targets_flat)
            
            # Backprop
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(data_loader)}], "
                      f"Loss: {running_loss/10:.4f}")
                running_loss = 0.0

    print("Training complete!")


if __name__ == "__main__":
    main()