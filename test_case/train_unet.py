#!/usr/bin/env python3
# Simple example to demonstrate how to do training with PyTorch using the ERA5TimeSeriesDataset class.

import os
import time
import argparse
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from ERA5TimeSeriesDataset import ERA5Dataset, PyTorchERA5Dataset

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
    input_vars = ['t2m', 'V500', 'U500', 'T500', 'Z500', 'Q500']
    target_vars = ['t2m', 'V500', 'U500', 'T500', 'Z500', 'Q500']

    train_start_year, train_end_year = 2000, 2017
    test_start_year, test_end_year = 2018, 2022

    # -----------------------------------------------------------------------
    # Create train, test, and validation datasets
    # -----------------------------------------------------------------------
    print("Loading datasets...")

    # 1) Training dataset: 2000 - 2017
    train_dataset = ERA5Dataset(
        data_path=data_path,
        start_year=train_start_year,
        end_year=train_end_year,
        input_vars=input_vars,
        target_vars=target_vars
    )
    train_dataset.normalize(mean_file='/glade/campaign/cisl/aiml/ksha/CREDIT/mean_6h_0.25deg.nc', std_file='/glade/campaign/cisl/aiml/ksha/CREDIT/std_6h_0.25deg.nc')
    train_pytorch = PyTorchERA5Dataset(train_dataset, forecast_step=1)
    train_loader = DataLoader(train_pytorch, batch_size=argv.batch_size, shuffle=True, pin_memory=True)

    # 2) Testing dataset: 2018 - 2022
    test_dataset = ERA5Dataset(
        data_path=data_path,
        start_year=test_start_year,
        end_year=test_end_year,
        input_vars=input_vars,
        target_vars=target_vars
    )
    test_dataset.normalize(mean_file='/glade/campaign/cisl/aiml/ksha/CREDIT/mean_6h_0.25deg.nc', std_file='/glade/campaign/cisl/aiml/ksha/CREDIT/std_6h_0.25deg.nc')
    test_pytorch = PyTorchERA5Dataset(test_dataset, forecast_step=1)
    test_loader = DataLoader(test_pytorch, batch_size=argv.batch_size, shuffle=False, pin_memory=True)


    print("Data loaded!")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples:  {len(test_loader.dataset)}")
    print ('-'*50)


    # --------------------------
    # Define the U-Net model using segmentation_models_pytorch
    ENCODER = 'resnet34'  # Encoder backbone
    ENCODER_WEIGHTS = 'imagenet'  # Pretrained weights
    #ENCODER_WEIGHTS = None  # No pretrained weights
    CLASSES = input_vars  # Number of output channels (same as input variables)
    ACTIVATION = None  # No activation for regression tasks

    # Create the U-Net model
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights="imagenet",
        decoder_attention_type="scse",
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
    criterion = torch.nn.L1Loss()
    #criterion = nn.SmoothL1Loss(beta=1.0)  # Huber Loss for robust training


    #optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # Adam optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)  

    # --------------------------
    # Training Loop
    print("Starting training loop...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(train_loader):
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
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevents exploding gradients

            optimizer.step()
            
            running_loss += loss.item()
            
            step_time = (time.time() - start_time)  # Compute elapsed time in milliseconds
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}, Time per step: {step_time:.2f} s.")

    # -----------------------------------------------------------------
    # Evaluation on Test (RMSE)
    # -----------------------------------------------------------------
    model.eval()
    test_rmse = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_rmse += rmse(outputs, targets).item()

    test_rmse /= len(test_loader)
    print(f"Test RMSE: {test_rmse:.4f}")


    print("Training complete!")

if __name__ == "__main__":
    main()
