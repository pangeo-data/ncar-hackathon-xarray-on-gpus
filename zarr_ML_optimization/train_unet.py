#!/usr/bin/env python3
"""
A U-Net benchmark using PyTorch and ERA5 surface variables
"""
import os
import time
import argparse
import logging

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

from trainer_utils import setup_logging, set_random_seeds, init_process_group, custom_loss


# --- Logger Setup ---
logger = logging.getLogger(__name__)


def main():
    num_epochs_default = 1
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
        help="Use synthetic data to skip loading ERA5 data (for benchmarking).",
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
    parser.add_argument(
        "--era5_path",
        type=str,
        help="Path to the ERA5 Zarr dataset.",
        default="/glade/derecho/scratch/negins/CREDIT_data/ERA5_mlevel_arXiv",
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
    # --------------------------

    setup_logging(world_rank=WORLD_RANK)
    logger.info("Strat training script!")
    logger.info(f"Distributed     : {distributed}")
    logger.info(f"Using DALI      : {use_dali}")
    logger.info(f"Synthetic data  : {use_synthetic}")

    # --------------------------
    # Read the ERA5 Zarr dataset
    data_path = argv.era5_path #"/glade/derecho/scratch/negins/CREDIT_data/ERA5_mlevel_arXiv"
    if use_dali:
        input_vars = ["combined"] * 6  # 6 input variables -- here for proof of concept, we combine all 6 input variables into one "combined" variable
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

    requested_workers = 1
    num_workers = min(
        requested_workers, 
        os.cpu_count() // 2,  # Safe default
        torch.cuda.device_count() * 4  # GPU-aware
    )

    if use_dali:
        # pipe_train = seqzarr_pipeline()
        # train_loader = DALIGenericIterator(
        #    pipelines=pipe_train, output_map=["input", "target"]
        # )
        source = SeqZarrSource(batch_size=batch_size)
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
                num_workers=num_workers,
                sampler=train_sampler,
                persistent_workers=True,
            )
        else:
            train_loader = DataLoader(
                train_pytorch,
                batch_size=batch_size,
                pin_memory=True,
                num_workers=num_workers,
                persistent_workers=True,
            )

    # --------------------------
    # 2) validation dataset
    # --------------------------
    if use_dali:
        source = SeqZarrSource(batch_size=batch_size)
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
                num_workers=num_workers,
                sampler=val_sampler,
                persistent_workers=True,
            )
        else:
            # val_loader = DataLoader(val_pytorch, batch_size=batch_size, pin_memory=True)
            val_loader = DataLoader(
                val_pytorch,
                batch_size=batch_size,
                pin_memory=True,
                num_workers=num_workers,
                persistent_workers=True,
            )

    
    if not use_dali:
        logger.info(f"Using PyTorch DataLoader (workers: {num_workers})")
        logger.info(f"Train samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples:  {len(val_loader.dataset)}")

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
        model = model.to(device)
        # Wrap the model with DDP
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[LOCAL_RANK],
            output_device=LOCAL_RANK,
            find_unused_parameters=True,
        )
        model = ddp_model.to(device)
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
    epoch_metrics_history = []

    logger.info("-" * 50)
    logger.info("Starting training loop...")

    for epoch in range(num_epochs):
        epoch_train_steps = 0
        epoch_val_steps = 0
        running_loss = 0.0
        epoch_train_losses = []

        epoch_start_time = time.time()

        model.train()
        
        for i, batch in enumerate(train_loader):
            start_time = time.time()  # Start time for the  train step

            if early_stop > 0 and i >= early_stop:
                break

            if len(batch) == 1:  # DALI
                inputs = batch[0]["input"].squeeze(dim=(0, 2))
                targets = batch[0]["target"].squeeze(dim=(0, 2))
            else:  # non-DALI
                inputs, targets = batch

            inputs, targets = inputs.to(device), targets.to(device)

            if not argv.notraining:  # training

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

                epoch_train_steps += 1

                if WORLD_RANK == 0:
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
                        f"Loss: {loss.item():.4f},"
                        f"RMSE: {loss_components['rmse']:.2f}, Std Diff: {loss_components['std_diff']:.2f},",
                        f"Time per training step: {step_train_time:.4f} sec.",
                    )

            else:  # Skip training for benchmarking purposes
                # Time should come out as 0.0
                torch.cuda.synchronize()
                step_train_time = time.time() - start_time
                epoch_train_steps += 1
                if WORLD_RANK == 0:
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
                        f"Time per training step: {step_train_time:.4f} sec."
                    )

        # End of training loop for this epoch
        stop_train_time = time.time()

        # reset the training loader for the next epoch
        if use_dali:
            train_loader.reset()

        # -----------------------------------------------------------------
        # Validation Loop
        # -----------------------------------------------------------------
        logger.info("-" * 50)
        logger.info("Starting validation loop...")
        model.eval()
        epoch_val_losses = []

        start_val_time = time.time()
        with torch.no_grad():

            # for i, (inputs, targets) in enumerate(val_loader):
            for i, batch in enumerate(val_loader):
                if early_stop > 0 and i >= early_stop:
                    logger.info("Skipping validation for benchmarking purposes.")
                    if use_dali:
                        val_loader.reset()
                    break

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
                    step_val_time = time.time() - step_val_start_time
                    epoch_val_steps += 1

                    if WORLD_RANK == 0:
                        print(
                            f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(val_loader)}], "
                            f"Validation Loss: {loss.item():.4f},"
                            f"RMSE: {loss_components['rmse']:.2f}, Std Diff: {loss_components['std_diff']:.2f},",
                            f"Time per validation step: {step_val_time:.4f} sec.",
                        )

                else:
                    torch.cuda.synchronize()
                    # Skip validation for benchmarking purposes
                    step_val_time = time.time() - step_val_start_time
                    epoch_val_steps += 1
                    if WORLD_RANK == 0:
                        print(
                            f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(val_loader)}], "
                            f"Time per validation step: {step_val_time:.4f} sec."
                        )

        stop_val_time = time.time()


        if use_dali:
            val_loader.reset()

        if WORLD_RANK == 0:
            epoch_time = time.time() - epoch_start_time
            val_time = stop_val_time - start_val_time
            train_time = stop_train_time - epoch_start_time

            # Throughput calculation
            num_train_batches_processed = epoch_train_steps
            num_val_batches_processed = epoch_val_steps
            
            total_samples_processed_epoch = (num_train_batches_processed + num_val_batches_processed) * batch_size * WORLD_SIZE
            print (f"Total samples processed in epoch: {total_samples_processed_epoch}")

            # calculate sample size in MB
            # each sample is snap of input variable and target variables
            sample_size = (
                inputs.element_size() * inputs.nelement()
                + targets.element_size() * targets.nelement()
            ) / batch_size
            sample_size_mb = sample_size / 1024 / 1024

            throughput_sps = total_samples_processed_epoch / epoch_time
            throughput_mbps = (total_samples_processed_epoch* sample_size_mb) / epoch_time

            current_epoch_metrics = {
                "epoch": epoch + 1,
                "epoch_time": epoch_time,
                "train_time": train_time,
                "val_time": val_time,
                "throughput_sps": throughput_sps,
                "throughput_mbps": throughput_mbps,
                "total_samples": total_samples_processed_epoch,
                "sample_size_mb": sample_size_mb,
            }

            epoch_metrics_history.append(current_epoch_metrics)


            # Logging output
            print("\n" + "-" * 60)
            print(f"Epoch [{epoch+1}/{num_epochs}] Summary")
            print(f"  Total Epoch Time     : {epoch_time:.2f} sec")
            print(f"  Training Time        : {train_time:.2f} sec")
            print(f"  Validation Time      : {val_time:.2f} sec")
            print(f"  Time/Train Step      : {train_time / len(train_loader):.4f} sec")
            print(f"  Time/Validation Step : {val_time / len(val_loader):.4f} sec")
            print(f"  WORLD_SIZE           : {WORLD_SIZE}")
            print(f"  Total Samples        : {total_samples_processed_epoch}")
            print(f"  Sample Size          : {sample_size_mb:.2f} MB")
            print(f"  Throughput (samples) : {throughput_sps:.2f} samples/sec")
            print(f"  Throughput (MB)      : {throughput_mbps:.2f} MB/sec")
            print("\n" + "-" * 60)


    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        logger.info("Destroyed process group!")
    
    # End of training loop for all epochs
    
    logger.info("Training completed.")
    
    if WORLD_RANK == 0:
        total_time = time.time() - training_start_time
        avg_epoch_wall_time = np.mean([m['epoch_time'] for m in epoch_metrics_history])
        avg_train_loop_time = np.mean([m['train_time'] for m in epoch_metrics_history])
        avg_val_loop_time = np.mean([m['val_time'] for m in epoch_metrics_history])
        avg_tput_sps = np.mean([m['throughput_sps'] for m in epoch_metrics_history])
        avg_tput_mbps = np.mean([m['throughput_mbps'] for m in epoch_metrics_history])
        avg_sample_size = np.mean([m['sample_size_mb'] for m in epoch_metrics_history])

        print("-" * 50)
        print ("Overall Training Summary (Averages Over All Epochs)")
        print(f"Total training time (sec)             : {total_time:.2f}")
        print(f"Average epoch wall time (sec)         : {avg_epoch_wall_time:.2f}")
        print(f"Average train loop time (sec)         : {avg_train_loop_time:.2f}")
        print(f"Average validation loop time (sec)    : {avg_val_loop_time:.2f}")
        print(f"Average throughput (samples/sec)      : {avg_tput_sps:.2f}")
        print(f"Average throughput (MB/sec)           : {avg_tput_mbps:.2f}")
        print(f"Average sample size (MB)              : {avg_sample_size:.2f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
