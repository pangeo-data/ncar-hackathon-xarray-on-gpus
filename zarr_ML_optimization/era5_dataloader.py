#!/usr/bin/env python3
"""
This module defines classes to handle ERA5 datasets stored in Zarr format,
including support for PyTorch DataLoader and NVIDIA DALI pipelines.

- ERA5Dataset: Load multi-year ERA5 data from Zarr stores. (No PyTorch dependency)
- PyTorchERA5Dataset: PyTorch-compatible wrapper for ERA5Dataset.

- SeqZarrSource: NVIDIA DALI-compatible external source for ERA5 Zarr data.
- seqzarr_pipeline: DALI pipeline for loading Zarr data using SeqZarrSource.

Example:
    python ERA5TimeSeriesDataset.py
    - Use the `--use-dali` flag to load data using DALI pipeline.
"""
import os
from contextlib import nullcontext

import numpy as np
import cupy as cp
import torch
import xarray as xr
import zarr

import nvidia.dali as dali
from nvidia.dali.pipeline import pipeline_def
from torch.utils.data import Dataset, DataLoader
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.fn as fn

class ERA5Dataset:
    """
    Load multiple years of ERA5 and forcing datasets from Zarr. 
    Each __getitem__(index) returns (input, target) as NumPy arrays.
    """

    def __init__(self, data_path, start_year, end_year, input_vars, target_vars=None,forecast_step=1, use_synthetic=False):
        """
        Initializes the dataset.

        Args:
            data_path (str): Path to the zarr store base....
            start_year (int): Start year for the dataset.
            end_year (int): End year for the dataset.
            input_vars (list): List of input variable names.
            target_vars (list, optional): List of target variable names. Defaults to input_vars.
        """
        self.data_path = data_path
        self.start_year = start_year
        self.end_year = end_year
        self.input_vars = input_vars
        self.target_vars = target_vars if target_vars is not None else input_vars
        self.normalized= False
        self.forecast_step = forecast_step
        self.use_synthetic = True if use_synthetic else False

        # load all zarr:
        self.dataset = self._load_data()
        self.ds_x, self.ds_y = self.fetch_timeseries(self.forecast_step)  # Precompute pairs
        self.length = self.ds_x.sizes['time']  # Update length based on valid pairs

    def _load_data(self):
        """Loads all zarr files into a dictionary keyed by year."""
        zarr_paths = []
        for year in range(self.start_year, self.end_year + 1):
            zarr_path = os.path.join(self.data_path, f"SixHourly_y_TOTAL_{year}-01-01_{year}-12-31_rechunked_uncompressed.zarr")
            if os.path.exists(zarr_path):
                zarr_paths.append(zarr_path)
            else:
                print (f"{zarr_path} does not exist for year {year}. Skipping...")
        ds = xr.open_mfdataset(zarr_paths, engine='zarr', consolidated=True, combine='by_coords')[self.input_vars]
        self.length = ds.sizes['time']
        return ds

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.length


    def fetch_timeseries(self, forecast_step=1):
        """
        Fetches the input and target timeseries data for a given forecast step.
        """
        ds_x = self.dataset.isel(time=slice(None, -forecast_step))
        ds_y = self.dataset.isel(time=slice(forecast_step, None))
        return ds_x, ds_y
    
    def normalize (self, mean_file=None, std_file=None):
        """
        Normalize the dataset using the mean and std files.
        """
        if mean_file is not None and std_file is not None:
            mean = xr.open_dataset(mean_file)
            std = xr.open_dataset(std_file)
        else:
            mean = self.dataset.mean(dim='time')
            std = self.dataset.std(dim='time')
        self.dataset = (self.dataset - mean) / std
        self.normalized = True

    def __repr__(self):
        """Returns a summary of all datasets loaded."""
        return self.dataset.__repr__()

    def __getitem__(self, index):
        """Enable direct indexing"""
        if self.use_synthetic:
            x_data = np.zeros([6, 640, 1280], dtype=np.float32)
            y_data = np.zeros([6, 640, 1280], dtype=np.float32)
        else:
            x_data = self.ds_x.isel(time=index).to_array().values
            y_data = self.ds_y.isel(time=index).to_array().values
        return (x_data, y_data)
 


class PyTorchERA5Dataset(Dataset):
    """
    Wraps the ERA5TimeSeriesDataset so it can be used in PyTorch DataLoader.
    """
    def __init__(self, era5_dataset, forecast_step=1):
        """
        era5_dataset (ERA5Dataset): An instance of the custom ERA5 dataset.
        forecast_step (int): The forecast step to use for fetching timeseries data.
        """
        self.era5_dataset = era5_dataset
        self.forecast_step = forecast_step
        self.ds_x, self.ds_y = self.era5_dataset.fetch_timeseries(forecast_step=self.forecast_step)
        self.use_synthetic = self.era5_dataset.use_synthetic
    
    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.ds_x.sizes['time']

    def __getitem__(self, index):
        """
        Returns a single sample (input, target) as PyTorch tensors.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: (input_tensor, target_tensor)
        """
        if self.use_synthetic:
            x_tensor = torch.zeros([6, 640, 1280], dtype=torch.float32)
            y_tensor = torch.zeros([6, 640, 1280], dtype=torch.float32)
        else:
            x_data = self.ds_x.isel(time=index).to_array().values
            y_data = self.ds_y.isel(time=index).to_array().values
            # Extract data at the given index
            x_data = self.ds_x.isel(time=index).to_array().values
            y_data = self.ds_y.isel(time=index).to_array().values
    
            # Convert to PyTorch tensors
            x_tensor = torch.from_numpy(x_data).float()
            y_tensor = torch.from_numpy(y_data).float()
    
        return x_tensor, y_tensor

    def __repr__(self):
        x_tensor, y_tensor = self[0]
        """Returns a summary of all datasets loaded."""
        return (
            f"PyTorchERA5Dataset(forecast_step={self.forecast_step}, "
            f"use_synthetic={self.use_synthetic}, "
            f"length={len(self)}, "
            f"input_tensor_shape={tuple(x_tensor.shape)}, "
            f"target_tensor_shape={tuple(y_tensor.shape)}, "
        )

class SeqZarrSource:
    """
    DALI Source for loading a zarr array.
    The arrays will be indexed along the first dimension (usually time).

    https://github.com/NVIDIA/modulus/blob/e6d7b02fb19ab9cdb3138de228ca3d6f0c99e7d1/examples/weather/unified_recipe/seq_zarr_datapipe.py#L186
    """

    def __init__(
        self,
        file_store: str = "/glade/derecho/scratch/negins/era5/rechunked_stacked_uncompressed_test.zarr",
        variables: list[str] = ["combined"],
        start_year: int = 2010,
        end_year: int = 2010,
        num_steps: int = 2,
        batch_size: int = 16,
        shuffle: bool = True,
        process_rank: int = 0,
        world_size: int = 1,
        batch: bool = True,
        gpu: bool = True,
    ):
        # Set up parameters
        self.file_store = file_store
        self.variables = variables
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch = batch
        self.gpu = gpu

        # Check if all zarr arrays have the same first dimension
        _zarr_dataset: zarr.Group = zarr.open(self.file_store, mode="r")
        self.first_dim: int = _zarr_dataset[variables[0]].shape[0]
        for variable in self.variables:
            if _zarr_dataset[variable].shape[0] != self.first_dim:
                raise ValueError("All zarr arrays must have the same first dimension.")

        # Get number of samples
        self.indices: np.ndarray = np.arange(
            batch_size
            * world_size
            * ((self.first_dim - self.num_steps) // batch_size // world_size)
        )
        self.indices: np.ndarray = np.array_split(self.indices, world_size)[
            process_rank
        ]

        # Get number of full batches, ignore possible last incomplete batch for now.
        self.num_batches: int = len(self.indices) // self.batch_size

        # Set up last epoch
        self.last_epoch = None

        # Set zarr dataset
        self.zarr_dataset = None

        # Set call
        if self.batch:
            self._call = self.__call__
            self.batch_mapping: np.ndarray = np.stack(
                np.array_split(
                    self.indices[
                        : len(self.indices) - len(self.indices) % self.batch_size
                    ],
                    self.batch_size,
                ),
                axis=1,
            )
        else:
            self._call = self._sample_call

        print (self.batch_mapping.shape)

    def __call__(self, index: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        with zarr.config.enable_gpu() if self.gpu else nullcontext():
            # Open Zarr dataset
            if self.zarr_dataset is None:
                self.zarr_dataset: zarr.Group = zarr.open(self.file_store, mode="r")

            #index: int = index[
            #    0
            #]  # turn [np.ndarray()] with one element to np.ndarray()
            index = int(index[0])

            if index > self.batch_mapping.shape[0]:
                raise StopIteration()

            # Get batch indices
            if self.gpu:
                self.batch_mapping = cp.asanyarray(self.batch_mapping)
            batch_idx: np.ndarray = self.batch_mapping[index]
            time_idx: np.ndarray = cp.concatenate(
                [idx + cp.arange(self.num_steps) for idx in batch_idx]
            )
            # print(time_idx)

            # Get data
            data = []

            # Get slices
            for i, variable in enumerate(self.variables):
                batch_data = self.zarr_dataset[variable][time_idx.tolist()]
                data.append(
                    cp.reshape(
                        batch_data,
                        (self.batch_size, self.num_steps, *batch_data.shape[1:]),
                    )
                )
            # assert len(data) == 6  # number of variables
            # assert data[0].shape == (16, 2, 640, 1280)  # BTHW

            # Stack variables along channel dimension, and split into two along timestep dim
            data_stack = cp.stack(data, axis=2)
            # assert data_stack.shape == (16, 2, 6, 640, 1280)  # BTCHW
            data_x = data_stack[:, 0, :, :, :]
            # assert data_x.shape == (16, 6, 640, 1280)  # BCHW
            data_y = data_stack[:, 1, :, :, :]
            # assert data_y.shape == (16, 6, 640, 1280)  # BCHW

            # Return list to satisfy batch_processing=True
            return [data_x], [data_y]


    def __len__(self):
        if self.batch:
            print(f"Batch mapping shape: {self.batch_mapping.shape}")
            return len(self.batch_mapping)
        else:
            return len(self.indices)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  file_store={self.file_store!r},\n"
            f"  variables={self.variables},\n"
            f"  num_steps={self.num_steps},\n"
            f"  batch_size={self.batch_size},\n"
            f"  shuffle={self.shuffle},\n"
            f"  batch={self.batch},\n"
            f"  gpu={self.gpu},\n"
            f"  first_dim={self.first_dim},\n"
            f"  total_samples={len(self.indices)},\n"
            f"  num_batches={self.num_batches}\n"
            f")"
        )


def build_seqzarr_pipeline(source: SeqZarrSource, batch_size: int = 16):
    """
    Build the DALI pipeline for loading Zarr data.
    """
    @pipeline_def(
        batch_size=4,
        num_threads=2,
        prefetch_queue_depth=2,
        py_num_workers=2,
        device_id=0,
        py_start_method="spawn",
    )

    def seqzarr_pipeline():
        """
        Pipeline to load Zarr stores via a DALI External Source operator.
        """
        # Zarr source
        source = SeqZarrSource(batch_size=16)
        print (source)
        print ("shape of this source:", source.__len__())

        # generate indexes for the external source
        def index_generator(idx: int) -> np.ndarray:
            return np.array([idx])

        indexes = dali.fn.external_source(
            source=index_generator,
            dtype=dali.types.INT64,
            device="gpu" if source.gpu else "cpu",
            batch=True,
        )

        print (indexes)

        # Use DALI to read current batch from SeqZarrSource
        data_x, data_y = dali.fn.python_function(
            indexes,
            function=source,
            batch_processing=True,
            num_outputs=2,
            device="gpu" if source.gpu else "cpu",
        )

        #data_x = data_x.squeeze(0).squeeze(1)
        #data_y = data_y.squeeze(0).squeeze(1)

        data_x = fn.reshape(data_x, shape=[16,6,640,1280])
        data_y = fn.reshape(data_y, shape=[16,6,640,1280])

        # if self.device.type == "cuda":
        # Move tensors to GPU as external_source won't do that automatically
        if not source.gpu:
            data_x = data_x.gpu()
        data_y = data_y.gpu()
    

        # Set outputs
        return data_x, data_y
    return seqzarr_pipeline()

# -------------------------------------------------#
# ----------------- Example usage -----------------#
# -------------------------------------------------#
if __name__ == "__main__":
    import argparse

    # Set up simple argument parser
    parser = argparse.ArgumentParser(description="ERA5 Data Loader...")
    parser.add_argument(
        "--use-dali",
        action="store_true",
        help="Use DALI pipeline instead of standard loading"
    )
    args = parser.parse_args()

    # Hardcoded parameters (as in original code)
    data_path = "/glade/derecho/scratch/ksha/CREDIT_data/ERA5_mlevel_arXiv"
    input_vars = ['t2m', 'V500', 'U500', 'T500', 'Z500', 'Q500']
    target_vars = ['t2m', 'V500', 'U500', 'T500', 'Z500', 'Q500']
    start_year = 2010
    end_year = 2010

    if not args.use_dali:
        # Standard dataset loading
        train_dataset = ERA5Dataset(
            data_path=data_path,
            start_year=start_year,
            end_year=end_year,
            input_vars=input_vars,
            target_vars=target_vars
        )
        print(train_dataset)
        train_pytorch = PyTorchERA5Dataset(
            train_dataset,
            forecast_step=1
        )
        print(train_pytorch)

        # Example of using PyTorch DataLoader
        train_loader = DataLoader(train_pytorch, batch_size=16, pin_memory=True, shuffle=True)
        print (f"Number of batches: {len(train_loader)}")
        print (f"Batch size: {train_loader.batch_size}")

        for i, batch in enumerate(train_loader):
            inputs, targets = batch

            print(f"Batch {i+1}: inputs shape = {inputs.shape}, targets shape = {targets.shape}")

            sample_size_bytes = (
                inputs.element_size() * inputs.nelement() +
                targets.element_size() * targets.nelement()
            )
            sample_size_mb = sample_size_bytes / 1024 / 1024 / inputs.shape[0]  # per sample
            print(f"Estimated sample size: {sample_size_mb:.2f} MB")

            print(f"Total samples in dataset: {len(train_loader)}")

            break

    else:
        # DALI pipeline loading
        source = SeqZarrSource(batch_size=16)
        print ("...shape of this source:", source.__len__())
        pipe = build_seqzarr_pipeline(source=source)
        pipe.build()
        


        #pipe = seqzarr_pipeline()
        train_loader = DALIGenericIterator(
            pipelines=pipe,
            output_map=["input", "target"],
            auto_reset=True,
            last_batch_padded=False,
            #fill_last_batch=False,
            #size = -1,
        )
        print (f"Number of batches: {len(train_loader)}")