#!/usr/bin/env python3
# Simple example to demonstrate how to do training with PyTorch using the ERA5TimeSeriesDataset class.

import os
import time
import argparse

import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import segmentation_models_pytorch as smp

from ERA5TimeSeriesDataset import ERA5Dataset, PyTorchERA5Dataset
from ERA5TimeSeriesDataset import ERA5DALIDataLoader  # Import the DALI-based loader
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from ERA5TimeSeriesDataset import ERA5Dataset, PyTorchERA5Dataset


def set_random_seeds(random_seed=0):
    """
    Set random seeds for reproducibility.
    """
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

@pipeline_def
def dali_era5_pipeline(input_data):
    """
    NVIDIA DALI pipeline for processing ERA5 dataset.

    Args:
        input_data: ERA5 dataset (NumPy arrays)

    Returns:
        Processed tensors for model training
    """

    # Load external source (ERA5 Data)
    input_data, target_data = fn.external_source(source=input_data, batch=True, num_outputs=2, dtype=types.FLOAT)

    return input_data, target_data


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
    parser.add_argument(
        "--distributed",
        action='store_true',
        help="Use distributed data parallel (DDP).",
    )


    argv = parser.parse_args()
    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    learning_rate = argv.learning_rate
    distributed = argv.distributed

    # Set random seeds for reproducibility!
    random_seed = 0
    set_random_seeds(random_seed=random_seed)

    # --------------------------
    # Distributed setup
    if distributed:
        try:
            # support for different flavors of MPI
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            shmem_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)                                                            
        
            LOCAL_RANK = shmem_comm.Get_rank()
            WORLD_SIZE = comm.Get_size()
            WORLD_RANK = comm.Get_rank()
        
            os.environ['MASTER_ADDR'] = comm.bcast( socket.gethostbyname( socket.gethostname() ), root=0 )
            os.environ['MASTER_PORT'] = '1234'
        
            if "MASTER_ADDR" not in os.environ:
                os.environ['MASTER_ADDR'] = comm.bcast( socket.gethostbyname( socket.gethostname() ), root=0 )
            if "MASTER_PORT" not in os.environ:
                os.environ['MASTER_PORT'] = str(np.random.randint(1000,8000))
        except:
            if "LOCAL_RANK" in os.environ:
                # Environment variables set by torch.distributed.launch or torchrun
                LOCAL_RANK = int(os.environ["LOCAL_RANK"])
                WORLD_SIZE = int(os.environ["WORLD_SIZE"])
                WORLD_RANK = int(os.environ["RANK"])
            elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
                # Environment variables set by mpirun
                LOCAL_RANK = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
                WORLD_SIZE = int(os.environ["OMPI_COMM_WORLD_SIZE"])
                WORLD_RANK = int(os.environ["OMPI_COMM_WORLD_RANK"])
            elif "PMI_RANK" in os.environ:
                # Environment variables set by cray-mpich
                LOCAL_RANK = int(os.environ["PMI_LOCAL_RANK"])
                WORLD_SIZE = int(os.environ["PMI_SIZE"])
                WORLD_RANK = int(os.environ["PMI_RANK"])
            else:
                import sys
                sys.exit("Can't find the evironment variables for local rank!")
    else:
        # for running without torchrun
        LOCAL_RANK = 0
        WORLD_SIZE = 1
        WORLD_RANK = 0

    if WORLD_RANK == 0:
        print ('----------------------')
        print ('LOCAL_RANK  : ', LOCAL_RANK)
        print ('WORLD_SIZE  : ', WORLD_SIZE)
        print ('WORLD_RANK  : ', WORLD_RANK)
        print("cuda device : ", torch.cuda.device_count())
        print("pytorch version : ", torch.__version__)
        print("nccl version : ", torch.cuda.nccl.version())
        print("torch config : ", torch.__config__.show())
        print(torch.__config__.parallel_info())
        print("----------------------")    

    # ---------------------
    if distributed:
        torch.distributed.init_process_group(
            backend="nccl", rank=WORLD_RANK, world_size=WORLD_SIZE
        )

    # --------------------------
    # Initialize the ERA5 Zarr dataset
    data_path = "/glade/derecho/scratch/ksha/CREDIT_data/ERA5_mlevel_arXiv"
    input_vars = ['t2m','V500', 'U500', 'T500', 'Z500', 'Q500'] # 6 input variables
    target_vars = ['t2m'] # Predict temperature only for now!!!!

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
    mean_file = '/glade/derecho/scratch/negins/hackathon-files/mean_6h_0.25deg.nc' # pre-computed mean file for normalization -- copied over from /glade/campaign/cisl/aiml/ksha/CREDIT/
    std_file = '/glade/derecho/scratch/negins/hackathon-files/std_6h_0.25deg.nc'  # pre-computed std file for normalization -- copied over from /glade/campaign/cisl/aiml/ksha/CREDIT/
    train_dataset.normalize(mean_file=mean_file, std_file=std_file)
    train_pytorch = PyTorchERA5Dataset(train_dataset, forecast_step=1)
    if distributed:
        train_sampler = DistributedSampler(dataset=train_pytorch, shuffle=False)
        train_loader = DataLoader(train_pytorch, batch_size=batch_size, pin_memory=True, sampler=train_sampler)
    else:
        train_loader = DataLoader(train_pytorch, batch_size=batch_size, pin_memory=True, shuffle=True)

    # 2) valing dataset: 2018 - 2022
    val_dataset = ERA5Dataset(
        data_path=data_path,
        start_year=val_start_year,
        end_year=val_end_year,
        input_vars=input_vars,
        target_vars=target_vars
    )
    val_dataset.normalize(mean_file=mean_file, std_file=std_file)
    val_pytorch = PyTorchERA5Dataset(val_dataset, forecast_step=1)
    if distributed:
        val_sampler = DistributedSampler(dataset=val_pytorch, shuffle=False)
        val_loader = DataLoader(val_pytorch, batch_size=batch_size, pin_memory=True, sampler = val_sampler)  
    else:
        val_loader = DataLoader(val_pytorch, batch_size=batch_size, pin_memory=True, shuffle=True)  

    print("Data loaded!")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples:  {len(val_loader.dataset)}")
    print ('-'*50)

    # --------------------------
    use_dali = True
    if use_dali:
        print("Using NVIDIA DALI for data loading...")
        train_dali_loader = ERA5DALIDataLoader(train_dataset, batch_size=batch_size)
        val_dali_loader = ERA5DALIDataLoader(val_dataset, batch_size=batch_size)

        train_pipeline = dali_era5_pipeline(batch_size=batch_size, num_threads=4, device_id=LOCAL_RANK, input_data=train_dali_loader)
        val_pipeline = dali_era5_pipeline(batch_size=batch_size, num_threads=4, device_id=LOCAL_RANK, input_data=val_dali_loader)

        train_loader = DALIGenericIterator(pipelines=[train_pipeline], output_map=["input", "target"], size=len(train_dataset), auto_reset=True)
        val_loader = DALIGenericIterator(pipelines=[val_pipeline], output_map=["input", "target"], size=len(val_dataset), auto_reset=True)
        print("DALI data loaders created!")


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
    if distributed:
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda:{}".format(LOCAL_RANK))                                                           
        print ("device:", device, "world_rank:", WORLD_RANK, "local_rank:", LOCAL_RANK)
        model = model.to(LOCAL_RANK)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    if distributed:
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True
        )
        model = ddp_model

    torch.backends.cudnn.benchmark = True

    # --------------------------
    # Define the loss function and optimizer
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()
    #criterion = nn.SmoothL1Loss(beta=1.0)  # Huber Loss for robust training


    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)  

    # --------------------------
    # Training Loop
    print("Starting training loop...")


    training_start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        epoch_train_losses = []  # Track losses for this epoch
        
        #for i, (inputs, targets) in enumerate(train_loader):
        for i, batch in enumerate(train_loader):
            inputs, targets = batch[0]["input"], batch[0]["target"] if use_dali else batch
            inputs, targets = inputs.to(device), targets.to(device)
            print (inputs)
            print (targets)

            start_time = time.time()  # Start time for the step

            # Move tensors to GPU if available
            #inputs = inputs.to(device)
            #targets = targets.to(device)
            
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
            torch.cuda.synchronize()
            
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

        print (f'Epoch [{epoch+1}/{num_epochs}], '
               f'Average Training Loss: {avg_train_metrics["loss"]:.4f}, '
               f'Average Training RMSE: {avg_train_metrics["rmse"]:.2f}°C')

        # -----------------------------------------------------------------
        # Validation Loop
        # -----------------------------------------------------------------
        model.eval()
        epoch_val_losses = []

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss, loss_components = custom_loss(outputs, targets)
                epoch_val_losses.append(loss_components)
                torch.cuda.synchronize()


        # Calculate average validation metrics
        avg_val_metrics = {
            'loss': sum(entry['total'] for entry in epoch_val_losses) / len(epoch_val_losses),
            'rmse': sum(entry['rmse'] for entry in epoch_val_losses) / len(epoch_val_losses)
        }

        print (f'Epoch [{epoch+1}/{num_epochs}], '
               f'Average Validation Loss: {avg_val_metrics["loss"]:.4f}, '
               f'Average Validation RMSE: {avg_val_metrics["rmse"]:.2f}°C')

        epoch_time = (time.time() - epoch_start_time)

        print(f'Epoch [{epoch+1}/{num_epochs}], Time this epoch: {epoch_time:.2f} seconds')

        # save snapshot of the model
        if WORLD_RANK==0:
            # Create directory for saved models
            model_dir = "./saved_models"
            os.makedirs(model_dir, exist_ok=True)
            checkpoint_path = os.path.join(model_dir, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_metrics,
                'val_loss': avg_val_metrics,
            }, checkpoint_path)

            print(f"Saved model checkpoint to {checkpoint_path}!")

        if distributed:
            total_samples = len(train_loader.dataset) * batch_size * WORLD_SIZE * num_epochs
        else:
            total_samples = len(train_loader.dataset) * batch_size * num_epochs

        if WORLD_RANK == 0:
            print(f"Training samples processed this epoch: {total_samples}")
            print(f"Average Throughput: {total_samples / epoch_time:.2f} samples/sec.")

    total_time = (time.time() - training_start_time) 
    print(f"Total training time: {total_time:.2f} seconds!")
    print ('-'*50)

if __name__ == "__main__":
    main()