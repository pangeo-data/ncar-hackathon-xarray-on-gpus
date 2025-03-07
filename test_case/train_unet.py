#!/usr/bin/env python3
"""
Train a U-Net model using PyTorch and ERA5 surface variables.
"""

import os
import time
import argparse

import xarray as xr
import numpy as np

import itertools 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import segmentation_models_pytorch as smp
from ERA5TimeSeriesDataset import ERA5Dataset, PyTorchERA5Dataset


def set_random_seeds(random_seed=0):
    """
    Set random seeds for reproducibility.
    """
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

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

def measure_gpu_throughput(model, inputs, batch_size):
    inputs = inputs.to('cuda')
    model = model.to('cuda')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for i in range(0, inputs.size(0), batch_size):
            output = model(inputs[i:i + batch_size])
    end.record()
    torch.cuda.synchronize()
    latency = start.elapsed_time(end)
    throughput = inputs.size(0) * batch_size / latency
    return throughput

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
    parser.add_argument(
        "--skip-training",
        action='store_true',
        help="Skip training for benchmarking purposes.",
        dest='notraining',
    )
    parser.add_argument(
        "--synth",
        "--use-synthetic",
        action='store_true',
        help="Use synthetic data to skip loading ERA5 data.",
    )


    argv = parser.parse_args()
    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    learning_rate = argv.learning_rate
    distributed = argv.distributed
    use_synthetic = argv.synth

    if use_synthetic:
        print("Using synthetic data for training!")
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
    target_vars = ['t2m','V500', 'U500', 'T500', 'Z500', 'Q500'] # Predict all 6 variables

    train_start_year, train_end_year = 2013, 2014
    val_start_year, val_end_year = 2018, 2018

    # -----------------------------------------------------------------------
    # Create train and validation datasets
    # -----------------------------------------------------------------------
    # 1) Training dataset
    train_dataset = ERA5Dataset(
        data_path=data_path,
        start_year=train_start_year,
        end_year=train_end_year,
        input_vars=input_vars,
        target_vars=target_vars,
        forecast_step=1,
        use_synthetic=use_synthetic
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

    # --------------------------
    # 2) validation dataset
    # --------------------------
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


    if WORLD_RANK == 0:
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

    # --------------------------
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
    criterion = torch.nn.L1Loss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)  

    # --------------------------
    # Training Loop

    training_start_time = time.time()

    epoch_total_times = []
    epoch_train_times = []
    epoch_val_times = []

    throughput_samples = []
    throughput_mb = []
    
    for epoch in range(num_epochs):
        
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        epoch_train_losses = []  # Track losses for this epoch
        
        #for i, (inputs, targets) in enumerate(train_loader):
        for i, (inputs, targets) in itertools.islice(enumerate(val_loader), 2):

            start_time = time.time()  # Start time for the step

            if not argv.notraining:
                # Move tensors to GPU if available
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                
                # Compute loss
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
            
            else:
                # Skip training for benchmarking purposes
                # Time should come out as 0.0
                step_time = (time.time() - start_time) 
                print (f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
                    f"Time per training step: {step_time:.4f} sec.")
    
        stop_train_time = time.time()

        if not argv.notraining:
        # Calculate average training metrics for this epoch for training cases!
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

        start_val_time = time.time()
        with torch.no_grad():
            #for i, (inputs, targets) in enumerate(val_loader):
            for i, (inputs, targets) in itertools.islice(enumerate(val_loader), 2):

                step_start_time = time.time()
                if not argv.notraining:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss, loss_components = custom_loss(outputs, targets)
                    epoch_val_losses.append(loss_components)
                    torch.cuda.synchronize()
                    step_val_time = (time.time() - step_start_time)  # Compute elapsed time in milliseconds
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(val_loader)}], "
                        f"Validation Loss: {loss.item():.4f},"
                        f"RMSE: {loss_components['rmse']:.2f}, Std Diff: {loss_components['std_diff']:.2f}",
                        f"Time per validation step: {step_val_time:.4f} sec.")
                
                else:
                    # Skip validation for benchmarking purposes

                    step_val_time = (time.time() - step_start_time) 
                    print (f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(val_loader)}], "
                        f"Time per validation step: {step_val_time:.4f} sec.")

        
        torch.cuda.synchronize()
        stop_val_time = time.time()

        if not argv.notraining:
            # Calculate average validation metrics
            if distributed: 
                torch.distributed.barrier()
            avg_val_metrics = {
                'loss': sum(entry['total'] for entry in epoch_val_losses) / len(epoch_val_losses),
                'rmse': sum(entry['rmse'] for entry in epoch_val_losses) / len(epoch_val_losses)
            }

            print (f'Epoch [{epoch+1}/{num_epochs}], '
                f'Average Validation Loss: {avg_val_metrics["loss"]:.4f}, '
                f'Average Validation RMSE: {avg_val_metrics["rmse"]:.2f}°C')

        
        # save snapshot of the model
        if WORLD_RANK==0:
            if not argv.notraining:
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

        if WORLD_RANK == 0:
            epoch_time = (time.time() - epoch_start_time)
            val_time = (stop_val_time - start_val_time)
            train_time = (stop_train_time - epoch_start_time) 
            print(f'Epoch [{epoch+1}/{num_epochs}], Time this epoch: {epoch_time:.2f} seconds')
            print(f'Epoch [{epoch+1}/{num_epochs}], Training time this epoch: {train_time:.2f} seconds')
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation time this epoch: {val_time:.2f} seconds')
            print(f'Epoch [{epoch+1}/{num_epochs}], Time per training step   : {train_time/ len(train_loader):.2f} seconds')
            print(f'Epoch [{epoch+1}/{num_epochs}], Time per validation step : {val_time/ len(val_loader):.2f} seconds')
            
            if distributed:
                #print ("len(val_loader.dataset) : ", len(val_loader.dataset))
                total_samples = len(val_loader) * WORLD_SIZE
            else:
                total_samples = len(val_loader)
            

            print(f"Samples processed this epoch: {total_samples}")
            print(f"Average throughput this epoch: {total_samples / val_time:.2f} samples/sec.")

            #calculate sample size in MB
            # each sample is snap of input variable and target variables 
            sample_size = (inputs.element_size() * inputs.nelement() + targets.element_size() * targets.nelement())/batch_size
            sample_size_mb =  sample_size / 1024 / 1024

            print(f"Sample size: {sample_size_mb:.2f} MB")
            print(f"Average throughput per MB: {(total_samples * sample_size_mb)/ val_time:.2f} MB/sec.")

            epoch_total_times.append(epoch_time)
            epoch_train_times.append(train_time)
            epoch_val_times.append(val_time)
            throughput_samples.append(total_samples / val_time)
            throughput_mb.append((total_samples * sample_size_mb)/ val_time)

    if WORLD_RANK == 0:
        print ('-'*50)
        print ('Training completed!')
        total_time = (time.time() - training_start_time) 
        print(f"Total training time: {total_time:.2f} seconds!")
        print(f"Average time per epoch: {np.mean(epoch_total_times):.2f} seconds")
        print(f"Average training time per epoch: {np.mean(epoch_train_times):.2f} seconds")
        print(f"Average validation time per epoch: {np.mean(epoch_val_times):.2f} seconds")
        print(f"Average throughput per epoch: {np.mean(throughput_samples):.2f} samples/sec.")
        print(f"Average throughput per epoch: {np.mean(throughput_mb):.2f} MB/sec.")
        print ('-'*50)

if __name__ == "__main__":
    main()