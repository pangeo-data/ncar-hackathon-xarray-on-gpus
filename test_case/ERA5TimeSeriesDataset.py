#!/usr/bin/env python3
"""
This dummy class is created for handling ERA-5 organized by year in Zarr format.
This file contains two classes:
1. ERA5Dataset: A custom dataset class to load multiple years of ERA5 data (1 zarr store per year -- but why?). (No PyTorch dependency)
2. PyTorchERA5Dataset: A wrapper class to use the custom dataset in PyTorch DataLoader.
"""
import os

import numpy as np
import torch
import xarray as xr
import nvidia.dali as dali
from torch.utils.data import Dataset
import zarr

Tensor = torch.Tensor


class ERA5Dataset:
    """
    Load multiple years of ERA5 and forcing datasets from Zarr. 
    Each __getitem__(index) returns (input, target) as NumPy arrays.
    """

    def __init__(self, data_path, start_year, end_year, input_vars, target_vars=None,forecast_step=1):
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

        # load all zarr:
        self.dataset = self._load_data()
        self.ds_x, self.ds_y = self.fetch_timeseries(self.forecast_step)  # Precompute pairs
        self.length = self.ds_x.sizes['time']  # Update length based on valid pairs

    def _load_data(self):
        """Loads all zarr files into a dictionary keyed by year."""
        zarr_paths = []
        for year in range(self.start_year, self.end_year + 1):
            zarr_path = os.path.join(self.data_path, f"SixHourly_y_TOTAL_{year}-01-01_{year}-12-31_staged.zarr")
            if os.path.exists(zarr_path):
                zarr_paths.append(zarr_path)
            else:
                print(f"Data for year {year} not found!!!")
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
        # Extract data at the given index
        x_data = self.ds_x.isel(time=index).to_array().values
        y_data = self.ds_y.isel(time=index).to_array().values

        # Convert to PyTorch tensors
        x_tensor = torch.from_numpy(x_data).float()
        y_tensor = torch.from_numpy(y_data).float()

        return x_tensor, y_tensor


class SeqZarrSource:
    """
    DALI Source for loading a zarr array.
    The arrays will be indexed along the first dimension (usually time).

    https://github.com/NVIDIA/modulus/blob/e6d7b02fb19ab9cdb3138de228ca3d6f0c99e7d1/examples/weather/unified_recipe/seq_zarr_datapipe.py#L186
    """

    def __init__(
        self,
        file_store: str = "/glade/derecho/scratch/ksha/CREDIT_data/ERA5_mlevel_arXiv/SixHourly_y_TOTAL_2022-01-01_2022-12-31_staged.zarr/",  #: fsspec.mapping.FSMap,
        variables: list[str] = ["t2m", "V500", "U500", "T500", "Z500", "Q500"],
        num_steps: int = 2,
        batch_size: int = 1,
        shuffle: bool = True,
        process_rank: int = 0,
        world_size: int = 1,
        batch: bool = True,
    ):
        # Set up parameters
        self.file_store = file_store
        self.variables = variables
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch = batch

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

    def __call__(
        self,
        sample_info: dali.types.BatchInfo,
    ) -> tuple[Tensor, Tensor, np.ndarray, np.ndarray, np.ndarray]:
        # Open Zarr dataset
        if self.zarr_dataset is None:
            self.zarr_dataset: zarr.Group = zarr.open(self.file_store, mode="r")

        if sample_info >= self.batch_mapping.shape[0]:
            raise StopIteration()

        # Get batch indices
        batch_idx: np.ndarray = self.batch_mapping[sample_info]
        time_idx: np.ndarray = np.concatenate(
            [idx + np.arange(self.num_steps) for idx in batch_idx]
        )

        # Get data
        data = []

        # Get slices
        for i, variable in enumerate(self.variables):
            batch_data = self.zarr_dataset[variable][time_idx]
            data.append(
                np.reshape(
                    batch_data, (self.batch_size, self.num_steps, *batch_data.shape[1:])
                )
            )

        return tuple(data)

    def __len__(self):
        if self.batch:
            return self.batch_mapping.shape[0] * self.batch_size
        else:
            return len(self.indices)


# %%
if __name__ == "__main__":

    ## Example usage of the ERA5TimeSeriesDataset class
    data_path = "/glade/derecho/scratch/ksha/CREDIT_data/ERA5_mlevel_arXiv"
    start_year = 1990
    end_year = 2010
    input_vars = ['t2m', 'V500', 'U500', 'T500', 'Z500', 'Q500']

    use_dali: bool = True
    if not use_dali:
        train_dataset = ERA5Dataset(
            data_path, start_year, end_year, input_vars=input_vars
        )
        print(train_dataset)
        forecast_step = 1
        ds_x, ds_y = train_dataset.fetch_timeseries(forecast_step=1)

        print(ds_x)

        # or use with data loader for pytorch as follows:
        # pytorch_dataset = PyTorchERA5Dataset(train_dataset)
        # data_loader = DataLoader(pytorch_dataset, batch_size=16, shuffle=True)
        # etc....

    else:  # use_dali is True
        pipe = dali.Pipeline(
            batch_size=16,
            num_threads=4,
            prefetch_queue_depth=4,
            py_num_workers=4,
            device_id=0,
            py_start_method="fork",
        )
        with pipe:
            # Zarr source
            source = SeqZarrSource()

            # Update length of dataset
            # self.total_length: int = len(source) // self.batch_size

            # Read current batch
            data = dali.fn.external_source(
                source,
                num_outputs=6,  # len(self.pipe_outputs),
                parallel=True,
                batch=True,
                prefetch_queue_depth=4,
                device="cpu",
            )

            # if self.device.type == "cuda":
            # Move tensors to GPU as external_source won't do that
            data = [d.gpu() for d in data]

            # Set outputs
            pipe.set_outputs(*data)

        pipe.build()
        arrays = pipe.run()
        print(arrays)
