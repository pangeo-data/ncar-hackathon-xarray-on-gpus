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

def custom_loss(predictions, targets, lambda_std=0.1):
    """
    Another custom loss function combining RMSE with standard deviation matching.

    The function handles two key aspects of the prediction quality:
    1. Accuracy: Through RMSE calculation
    2. Variability: Through standard deviation matching

    """
    # Calculate RMSE for prediction accuracy
    rmse_loss = torch.nn.functional.mse_loss(predictions, targets, reduction='mean').sqrt()

    # Calculate standard deviation component
    # We'll calculate std over the batch dimension (dim=0) and average over spatial dimensions
    # unbiased=False removes the Bessel correction and addresses the warning
    pred_std = torch.std(predictions.view(-1), unbiased=False)
    target_std = torch.std(targets.view(-1), unbiased=False)
    # Average the standard deviation differences across spatial dimensions
    std_loss = torch.mean(torch.abs(pred_std - target_std))

    # Combine the losses with the weighting factor
    total_loss = rmse_loss + lambda_std * std_loss

    # Store components for monitoring
    loss_components = {
        'rmse': rmse_loss.item(),
        'std_diff': std_loss.item(),
        'total': total_loss.item()
    }

    return total_loss, loss_components


def main():
    num_epochs_default = 2
    batch_size_default = 4
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
    # Initialize the ERA5 Zarr dataset
    data_path = "/glade/derecho/scratch/ksha/CREDIT_data/ERA5_mlevel_arXiv"
    input_vars = ['t2m']
    target_vars = ['t2m']

    train_start_year, train_end_year = 2013, 2014
    val_start_year, val_end_year = 2018, 2018

    # -----------------------------------------------------------------------
    # Create train, val, and validation datasets
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
    train_loader = DataLoader(train_pytorch, batch_size=batch_size, shuffle=True, pin_memory=True)

    # 2) valing dataset: 2018 - 2022
    val_dataset = ERA5Dataset(
        data_path=data_path,
        start_year=val_start_year,
        end_year=val_end_year,
        input_vars=input_vars,
        target_vars=target_vars
    )
    val_dataset.normalize(mean_file='/glade/campaign/cisl/aiml/ksha/CREDIT/mean_6h_0.25deg.nc', std_file='/glade/campaign/cisl/aiml/ksha/CREDIT/std_6h_0.25deg.nc')
    val_pytorch = PyTorchERA5Dataset(val_dataset, forecast_step=1)
    val_loader = DataLoader(val_pytorch, batch_size=batch_size, shuffle=False, pin_memory=True)


    print("Data loaded!")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples:  {len(val_loader.dataset)}")
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


    #optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)  

    # --------------------------
    # Training Loop
    print("Starting training loop...")

    # Create directory for saved models
    model_dir = "./saved_models"
    os.makedirs(model_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_train_losses = []  # Track losses for this epoch
        
        for i, (inputs, targets) in enumerate(train_loader):
            start_time = time.time()  # Start time for the step

            # Move tensors to GPU if available
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            #loss = criterion(outputs, targets)
            # Calculate loss using custom loss function
            loss, loss_components = custom_loss(outputs, targets)
            
            # Backprop
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevents exploding gradients

            optimizer.step()
            
            epoch_train_losses.append(loss_components)
            running_loss += loss.item()
            
            step_time = (time.time() - start_time)  # Compute elapsed time in milliseconds
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f},"
                  f"RMSE: {loss_components['rmse']:.2f}, Std Diff: {loss_components['std_diff']:.2f}",
                  f"Time per training step: {step_time:.4f} sec.")

        # Calculate average training metrics for this epoch
        avg_train_metrics = {
            'loss': sum(entry['total'] for entry in epoch_train_losses) / len(epoch_train_losses),
            'rmse': sum(entry['rmse'] for entry in epoch_train_losses) / len(epoch_train_losses)
        }

        # -----------------------------------------------------------------
        # Validation Loop
        # -----------------------------------------------------------------
        model.eval()
        #val_losses = []
        epoch_val_losses = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss, loss_components = custom_loss(outputs, targets)
                epoch_val_losses.append(loss.item())


        # Calculate average validation metrics
        avg_val_metrics = {
            'loss': sum(entry['rmse'] for entry in epoch_val_losses) / len(epoch_val_losses),
            'rmse': sum(entry['rmse'] for entry in epoch_val_losses) / len(epoch_val_losses)
        }

        print(f' Epoch [{epoch+1}/{num_epochs}], '
              f'Average Validation Loss: {avg_val_metrics["loss"]:.4f}, '
              f'Average Validation RMSE: {avg_val_metrics["rmse"][-1]:.2f}°C, '
              f'Average Validation Std Diff: {avg_val_metrics["std_diff"][-1]:.2f}')

                # Save model checkpoint
        checkpoint_path = os.path.join(model_dir, f"model_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'train_loss': train_metrics,
            'val_loss': val_metrics,
            'config': training_config
        }, checkpoint_path)
        
        logger.info(f"Saved model checkpoint to {checkpoint_path}")

    print("Training complete!")

if __name__ == "__main__":
    main()