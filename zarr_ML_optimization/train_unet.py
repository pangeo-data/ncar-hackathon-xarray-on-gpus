#!/usr/bin/env python3
"""
Train a U-Net model using PyTorch and ERA5 surface variables.
"""

import os
import time
import argparse
import logging
import socket

import xarray as xr
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import segmentation_models_pytorch as smp

from era5_dataloader import (
    ERA5Dataset,
    PyTorchERA5Dataset,
    build_seqzarr_pipeline,
    SeqZarrSource,
)


# --- Logger Setup ---
logger = logging.getLogger(__name__) # Get a logger specific to this module

def setup_logging(world_rank: int, level: int = logging.INFO) -> None:
    """Sets up basic logging. Logs only from rank 0."""
    if world_rank == 0:
        logging.basicConfig( # Configure the root logger
            level=level,
            format="%(asctime)s [%(levelname)s] : %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        logging.basicConfig(level=logging.CRITICAL + 1)

def set_random_seeds(random_seed=0):
    """
    Set random seeds for reproducibility.
    """
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(random_seed)


def custom_loss(predictions, targets, lambda_std=0.1):
    """
    Another custom loss function combining RMSE with standard deviation matching.

    The function handles two key aspects of the prediction quality:
    1. Accuracy: Through RMSE calculation
    2. Variability: Through standard deviation matching

    """
    # Calculate RMSE for prediction accuracy
    rmse_loss = torch.nn.functional.mse_loss(
        predictions, targets, reduction="mean"
    ).sqrt()

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
        "rmse": rmse_loss.item(),
        "std_diff": std_loss.item(),
        "total": total_loss.item(),
    }

    return total_loss, loss_components


def init_process_group(
    distributed: bool, backend: str = "nccl"
) -> tuple[int, int, int]:
    """
    Initialize the process group for distributed training.
    """
    if distributed:
        try:
            # support for different flavors of MPI
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            shmem_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)

            LOCAL_RANK = shmem_comm.Get_rank()
            WORLD_SIZE = comm.Get_size()
            WORLD_RANK = comm.Get_rank()

            os.environ["MASTER_ADDR"] = comm.bcast(
                socket.gethostbyname(socket.gethostname()), root=0
            )
            os.environ["MASTER_PORT"] = "1234"

            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = comm.bcast(
                    socket.gethostbyname(socket.gethostname()), root=0
                )
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = str(np.random.randint(1000, 8000))
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
                raise RuntimeError(
                    "Can't find the environment variables for local rank!"
                )
    else:
        # for running without torchrun or mpirun (i.e. ./train_unet.py)
        LOCAL_RANK = 0
        WORLD_SIZE = 1
        WORLD_RANK = 0

    if WORLD_RANK == 0:
        print("----------------------")
        print("LOCAL_RANK  : ", LOCAL_RANK)
        print("WORLD_SIZE  : ", WORLD_SIZE)
        print("WORLD_RANK  : ", WORLD_RANK)
        print("cuda device : ", torch.cuda.device_count())
        print("pytorch version : ", torch.__version__)
        print("nccl version : ", torch.cuda.nccl.version())
        print("torch config : ", torch.__config__.show())
        print(torch.__config__.parallel_info())
        print("----------------------")

    # ---------------------
    # Initialize distributed training
    if distributed:
        torch.distributed.init_process_group(
            backend=backend, rank=WORLD_RANK, world_size=WORLD_SIZE
        )
    return LOCAL_RANK, WORLD_SIZE, WORLD_RANK


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
        action="store_true",
        help="Use distributed data parallel (DDP).",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training for benchmarking purposes.",
        dest="notraining",
    )
    parser.add_argument(
        "--synth",
        "--use-synthetic",
        action="store_true",
        help="Use synthetic data to skip loading ERA5 data.",
    )
    parser.add_argument(
        "--use-dali",
        action="store_true",
        help="Use DALI pipeline instead of regular Pytorch Dataloader.",
    )
    parser.add_argument(
        "--early-stop",
        type=int,
        help="Stop after N steps per epoch for benchmarking purposes (default: 0, disabled).",
        default=0,
    )

    argv = parser.parse_args()
    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    learning_rate = argv.learning_rate
    distributed = argv.distributed
    use_synthetic = argv.synth
    use_dali = argv.use_dali
    early_stop = argv.early_stop


    # ---------------------------
    # Set random seeds for reproducibility!
    random_seed = 0
    set_random_seeds(random_seed=random_seed)

    # --------------------------
    # Initialize the process group for single or multi-GPU training
    LOCAL_RANK, WORLD_SIZE, WORLD_RANK = init_process_group(
        distributed=distributed, backend="nccl"
    )

    setup_logging(world_rank=WORLD_RANK)
    logger.info(f"Start training script with args: {argv}")
    logger.info(f"Using DALI: {use_dali}")
    logger.info(f"Using synthetic data: {use_synthetic}")
    logger.info(f"Distributed training: {distributed}")

    # --------------------------
    # Read the ERA5 Zarr dataset
    data_path = "/glade/derecho/scratch/negins/CREDIT_data/ERA5_mlevel_arXiv"
    if use_dali:
        input_vars = ["combined"] * 6  # 6 input variables
        target_vars = [
            "combined"
        ] * 6  # Stacked input variables into one "combined" variable
        # input_vars = ['t2m','V500', 'U500', 'T500', 'Z500', 'Q500'] # 6 input variables
        # target_vars = ['t2m','V500', 'U500', 'T500', 'Z500', 'Q500'] # Predict all 6 variables
    else:
        input_vars = [
            "t2m",
            "V500",
            "U500",
            "T500",
            "Z500",
            "Q500",
        ]  # 6 input variables
        target_vars = [
            "t2m",
            "V500",
            "U500",
            "T500",
            "Z500",
            "Q500",
        ]

    train_start_year, train_end_year = 2013, 2014
    val_start_year, val_end_year = 2018, 2018

    # -----------------------------------------------------------------------
    # Create train and validation datasets
    # -----------------------------------------------------------------------
    # 1) Training dataset

    if use_dali:
        # pipe_train = seqzarr_pipeline()
        # train_loader = DALIGenericIterator(
        #    pipelines=pipe_train, output_map=["input", "target"]
        # )
        source = SeqZarrSource(batch_size=16)
        print("...shape of this source:", source.__len__())
        pipe_train = build_seqzarr_pipeline(source=source)
        pipe_train.build()
        train_loader = DALIGenericIterator(
            pipelines=pipe_train, output_map=["input", "target"]
        )
        if distributed:
            raise NotImplementedError("DALI pipeline with distributed not working yet")
    elif not use_dali:
        train_dataset = ERA5Dataset(
            data_path=data_path,
            start_year=train_start_year,
            end_year=train_end_year,
            input_vars=input_vars,
            target_vars=target_vars,
            forecast_step=1,
            use_synthetic=use_synthetic,
        )

        # Normalize the dataset using pre-computed mean and std files
        mean_file = "/glade/derecho/scratch/negins/hackathon-files/mean_6h_0.25deg.nc"  # pre-computed mean file for normalization -- copied over from /glade/campaign/cisl/aiml/ksha/CREDIT/
        std_file = "/glade/derecho/scratch/negins/hackathon-files/std_6h_0.25deg.nc"  # pre-computed std file for normalization -- copied over from /glade/campaign/cisl/aiml/ksha/CREDIT/
        train_dataset.normalize(mean_file=mean_file, std_file=std_file)
        train_pytorch = PyTorchERA5Dataset(train_dataset, forecast_step=1)

        if distributed:
            train_sampler = DistributedSampler(dataset=train_pytorch, shuffle=False)
            train_loader = DataLoader(
                train_pytorch,
                batch_size=batch_size,
                pin_memory=True,
                num_workers=16,
                sampler=train_sampler,
                persistent_workers=True,
            )  # Use prefetching to speed up data loading
        else:
            train_loader = DataLoader(
                train_pytorch,
                batch_size=batch_size,
                pin_memory=True,
                num_workers=16,
                persistent_workers=True,
            )  # Use prefetching to speed up data loading

    # --------------------------
    # 2) validation dataset
    # --------------------------
    if use_dali:
        source = SeqZarrSource(batch_size=16)
        logger.info("...shape of this source:", source.__len__())
        pipe_val = build_seqzarr_pipeline(source=source)
        pipe_val.build()
        val_loader = DALIGenericIterator(
            pipelines=pipe_val, output_map=["input", "target"]
        )
        if distributed:
            raise NotImplementedError("DALI pipeline with distributed not working yet")
    elif not use_dali:  # classic pytorch dataset
        val_dataset = ERA5Dataset(
            data_path=data_path,
            start_year=val_start_year,
            end_year=val_end_year,
            input_vars=input_vars,
            target_vars=target_vars,
        )
        val_dataset.normalize(mean_file=mean_file, std_file=std_file)
        val_pytorch = PyTorchERA5Dataset(val_dataset, forecast_step=1)

        if distributed:
            val_sampler = DistributedSampler(dataset=val_pytorch, shuffle=False)
            val_loader = DataLoader(
                val_pytorch,
                batch_size=batch_size,
                pin_memory=True,
                num_workers=16,
                sampler=val_sampler,
                persistent_workers=True,
            )
        else:
            # val_loader = DataLoader(val_pytorch, batch_size=batch_size, pin_memory=True)
            val_loader = DataLoader(
                val_pytorch,
                batch_size=batch_size,
                pin_memory=True,
                num_workers=16,
                persistent_workers=True,
            )

    
    if not use_dali:
        logger.info("Using PyTorch DataLoader")
        logger.info(f"Train samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples:  {len(val_loader.dataset)}")
        logger.info("-" * 50)

    # --------------------------
    # Define the U-Net model using segmentation_models_pytorch
    ENCODER = "resnet18"  # Encoder backbone
    ENCODER_WEIGHTS = "imagenet"  # Pretrained weights
    # ENCODER_WEIGHTS = None  # No pretrained weights
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
    # Move the model to GPU
    if distributed:
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda:{}".format(LOCAL_RANK))
        print("device:", device, "world_rank:", WORLD_RANK, "local_rank:", LOCAL_RANK)
        model = model.to(device)
        # Wrap the model with DDP
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[LOCAL_RANK],
            output_device=LOCAL_RANK,
            find_unused_parameters=True,
        )
        model = ddp_model
        model = model.to(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    # --------------------------
    # Define the loss function and optimizer
    #criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )

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

        for i, batch in enumerate(train_loader):

            start_time = time.time()  # Start time for the step

            if early_stop > 0 and i >= early_stop:
                if use_dali:
                    pipe_train.reset()
                break

            if len(batch) == 1:  # DALI
                inputs = batch[0]["input"].squeeze(dim=(0, 2))
                targets = batch[0]["target"].squeeze(dim=(0, 2))
            else:  # non-DALI
                inputs, targets = batch

            if not argv.notraining:  # training

                # Move tensors to GPU if available
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)

                # Compute loss
                loss, loss_components = custom_loss(outputs, targets)

                # Backprop
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevents exploding gradients

                optimizer.step()
                torch.cuda.synchronize()

                epoch_train_losses.append(loss_components)
                running_loss += loss.item()

                step_train_time = (
                    time.time() - start_time
                )  # Compute elapsed time in milliseconds

                if WORLD_RANK == 0:
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
                        f"Loss: {loss.item():.4f},"
                        f"RMSE: {loss_components['rmse']:.2f}, Std Diff: {loss_components['std_diff']:.2f}",
                        f"Time per training step: {step_train_time:.4f} sec.",
                    )

            else:  # Skip training for benchmarking purposes
                # Time should come out as 0.0
                torch.cuda.synchronize()
                step_train_time = time.time() - start_time
                if WORLD_RANK == 0:
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
                        f"Time per training step: {step_time:.4f} sec."
                    )

        # End of training loop for this epoch
        stop_train_time = time.time()

        if early_stop > 0 and i >= early_stop:
            if use_dali:
                val_loader.reset()
            break

        if not argv.notraining:
            # Calculate average training metrics for this epoch for training cases!
            avg_train_metrics = {
                "loss": sum(entry["total"] for entry in epoch_train_losses)
                / len(epoch_train_losses),
                "rmse": sum(entry["rmse"] for entry in epoch_train_losses)
                / len(epoch_train_losses),
            }

            print(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f'Average Training Loss: {avg_train_metrics["loss"]:.4f}, '
                f'Average Training RMSE: {avg_train_metrics["rmse"]:.2f}°C.'
            )

        # reset the training loader for the next epoch
        if use_dali:
            train_loader.reset()

        # -----------------------------------------------------------------
        # Validation Loop
        # -----------------------------------------------------------------
        model.eval()
        epoch_val_losses = []

        start_val_time = time.time()
        with torch.no_grad():

            # for i, (inputs, targets) in enumerate(val_loader):
            for i, batch in enumerate(val_loader):
                step_val_start_time = time.time()  # Start time for the step

                if not argv.notraining:
                    if len(batch) == 1:  # DALI
                        inputs = batch[0]["input"].squeeze(dim=(0, 2))
                        targets = batch[0]["target"].squeeze(dim=(0, 2))
                    else:
                        inputs, targets = batch

                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss, loss_components = custom_loss(outputs, targets)
                    epoch_val_losses.append(loss_components)
                    torch.cuda.synchronize()
                    step_val_time = (
                        time.time() - step_val_start_time
                    )  # Compute elapsed time in milliseconds

                    if WORLD_RANK == 0:
                        print(
                            f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(val_loader)}], "
                            f"Validation Loss: {loss.item():.4f},"
                            f"RMSE: {loss_components['rmse']:.2f}, Std Diff: {loss_components['std_diff']:.2f}",
                            f"Time per validation step: {step_val_time:.4f} sec.",
                        )

                else:
                    # Skip validation for benchmarking purposes

                    step_val_time = time.time() - step_start_time
                    if WORLD_RANK == 0:
                        print(
                            f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(val_loader)}], "
                            f"Time per validation step: {step_val_time:.4f} sec."
                        )

        torch.cuda.synchronize()
        stop_val_time = time.time()

        if not argv.notraining:
            # Calculate average validation metrics
            if distributed:
                torch.distributed.barrier()
            avg_val_metrics = {
                "loss": sum(entry["total"] for entry in epoch_val_losses)
                / len(epoch_val_losses),
                "rmse": sum(entry["rmse"] for entry in epoch_val_losses)
                / len(epoch_val_losses),
            }

            if WORLD_RANK == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f'Average Validation Loss: {avg_val_metrics["loss"]:.4f}, '
                    f'Average Validation RMSE: {avg_val_metrics["rmse"]:.2f}°C.'
                )

        if use_dali:
            val_loader.reset()

        if WORLD_RANK == 0:
            epoch_time = time.time() - epoch_start_time
            val_time = stop_val_time - start_val_time
            train_time = stop_train_time - epoch_start_time
            print("-" * 50)
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Time this epoch: {epoch_time:.2f} seconds"
            )
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Training time this epoch: {train_time:.2f} seconds"
            )
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Validation time this epoch: {val_time:.2f} seconds"
            )
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Time per training step   : {train_time/ len(train_loader):.2f} seconds"
            )
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Time per validation step : {val_time/ len(val_loader):.2f} seconds"
            )

            if distributed:
                total_samples = (
                    (len(val_loader.dataset) + len(train_loader.dataset))
                    * batch_size
                    * WORLD_SIZE
                )
            else:
                total_samples = (
                    len(val_loader.dataset) + len(train_loader.dataset)
                ) * batch_size

            if early_stop >0 : 
                total_samples = (early_stop + early_stop) * batch_size
                if distributed:
                    total_samples = (
                        (early_stop + early_stop) * batch_size * WORLD_SIZE
                    )

            print(f"Samples processed this epoch: {total_samples}")
            print(
                f"Average throughput this epoch: {total_samples / epoch_time:.2f} samples/sec."
            )

            # calculate sample size in MB
            # each sample is snap of input variable and target variables
            sample_size = (
                inputs.element_size() * inputs.nelement()
                + targets.element_size() * targets.nelement()
            ) / batch_size
            sample_size_mb = sample_size / 1024 / 1024

            print(f"Sample size: {sample_size_mb:.2f} MB")
            print(
                f"Average throughput per MB: {(total_samples * sample_size_mb)/ val_time:.2f} MB/sec."
            )

            epoch_total_times.append(epoch_time)
            epoch_train_times.append(train_time)
            epoch_val_times.append(val_time)
            throughput_samples.append(total_samples / val_time)
            throughput_mb.append((total_samples * sample_size_mb) / val_time)

    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        logger.info("Destroyed process group.")
    
    # End of training loop for all epochs
    logger.info("Training completed.")

    if WORLD_RANK == 0:
        print("-" * 50)
        print("Training completed!")
        total_time = time.time() - training_start_time
        print(f"Total training time: {total_time:.2f} seconds!")
        print(f"Average time per epoch: {np.mean(epoch_total_times):.2f} seconds")
        print(
            f"Average training time per epoch: {np.mean(epoch_train_times):.2f} seconds"
        )
        print(
            f"Average validation time per epoch: {np.mean(epoch_val_times):.2f} seconds"
        )
        print(
            f"Average throughput per epoch: {np.mean(throughput_samples):.2f} samples/sec."
        )
        print(f"Average throughput per epoch: {np.mean(throughput_mb):.2f} MB/sec.")
        print("-" * 50)


if __name__ == "__main__":
    main()
