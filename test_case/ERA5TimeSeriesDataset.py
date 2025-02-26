#!/usr/bin/env python3
"""
This dummy class is created for handling ERA-5 organized by year in Zarr format.
This file contains two classes:
1. ERA5Dataset: A custom dataset class to load multiple years of ERA5 data (1 zarr store per year -- but why?). (No PyTorch dependency)
2. PyTorchERA5Dataset: A wrapper class to use the custom dataset in PyTorch DataLoader.
"""
import os
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np



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

    def __iter__(self):
        """Reset index for new iteration"""
        self.index = 0
        return self

    def __next__(self):
        """Return (input, target) pair with proper forecast offset"""
        if self.index < self.length:
            # Get input and target at current index
            x_data = self.ds_x.isel(time=self.index).to_array().values
            y_data = self.ds_y.isel(time=self.index).to_array().values
            self.index += 1
            return (x_data, y_data)
        raise StopIteration
 


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

# ---------------------------
# NVIDIA DALI Pipeline
# ---------------------------
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
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


# ---------------------------
# DALI Iterator Class
# ---------------------------

class ERA5DALIDataLoader:
    """
    # From ExternalInputIterator
    DALI-based DataLoader for handling dataset efficiently.
    """

    def __init__(self, dataset, batch_size=8, num_threads=4, device_id=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.device_id = device_id
        self.index = 0

    def __call__(self):
        """
        Generator function that provides batches of ERA5 data.
        """
        batch_x = []
        batch_y = []

        for _ in range(self.batch_size):
            if self.index >= len(self.dataset):
                self.index = 0  # Reset index if exceeding dataset length

            # This uses the __getitem__ method of the ERA5Dataset class
            x_data, y_data = self.dataset[self.index]
            batch_x.append(x_data)
            batch_y.append(y_data)
            #print (f"Batch {self.index}: {x_data.shape}, {y_data.shape}")   

            self.index += 1

        return np.stack(batch_x, axis=0), np.stack(batch_y, axis=0)
    
    def __repr__(self):
        return f"ERA5DALIDataLoader(batch_size={self.batch_size}, num_threads={self.num_threads}, device_id={self.device_id})"



if __name__ == "__main__":

    ## Example usage of the ERA5TimeSeriesDataset class
    data_path = "/glade/derecho/scratch/ksha/CREDIT_data/ERA5_mlevel_arXiv"
    start_year = 1990
    end_year = 2010
    input_vars = ['t2m', 'V500', 'U500', 'T500', 'Z500', 'Q500']

    train_dataset = ERA5Dataset(data_path, start_year, end_year, input_vars=input_vars)



    print (train_dataset)
    forecast_step = 1
    ds_x , ds_y = train_dataset.fetch_timeseries(forecast_step=1)

    print(ds_x)
    batch_size = 16
    print (ERA5DALIDataLoader(train_dataset, batch_size=batch_size))
    print ('***')
    pipeline = dali_era5_pipeline(batch_size=batch_size, num_threads=4, device_id=0, input_data=ERA5DALIDataLoader(train_dataset, batch_size=batch_size))

    # Create DALI PyTorch Iterator
    dali_loader = DALIGenericIterator(pipelines=[pipeline], output_map=["input", "target"], size=len(train_dataset), auto_reset=True)

    for i, data in enumerate(dali_loader):
        input_batch, target_batch = data[0]["input"], data[0]["target"]
        print(f"Batch {i+1}: Input shape {input_batch.shape}, Target shape {target_batch.shape}")
    # Alternatively, you can use the PyTorchERA5Dataset class directly:



    # or use with data loader for pytorch as follows:
    # pytorch_dataset = PyTorchERA5Dataset(train_dataset)
    # data_loader = DataLoader(pytorch_dataset, batch_size=16, shuffle=True)
    # etc....
