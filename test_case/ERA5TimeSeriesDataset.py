#!/usr/bin/env python3
"""
This dummy class is created for handling ERA-5 organized by year in Zarr format.
This file contains two classes:
1. ERA5Dataset: A custom dataset class to load multiple years of ERA5 data (1 zarr store per year -- but why?). (No PyTorch dependency)
2. PyTorchERA5Dataset: A wrapper class to use the custom dataset in PyTorch DataLoader.
"""
import os

import torch
import xarray as xr
from torch.utils.data import Dataset


def _load_data(
    data_path: str, start_year: int, end_year: int, input_vars: list[str]
) -> xr.Dataset:
    """Loads all zarr files into a dictionary keyed by year."""
    zarr_paths: list[str] = []
    for year in range(start_year, end_year + 1):
        zarr_path: str = os.path.join(
            data_path, f"SixHourly_y_TOTAL_{year}-01-01_{year}-12-31_staged.zarr"
        )
        if os.path.exists(zarr_path):
            zarr_paths.append(zarr_path)
        else:
            print(f"Data for year {year} not found!!!")
    ds: xr.Dataset = xr.open_mfdataset(
        zarr_paths, engine="zarr", consolidated=True, combine="by_coords"
    )[input_vars]
    # self.length = ds.sizes["time"]
    return ds


# %%
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
        self.dataset = _load_data(
            data_path=self.data_path,
            start_year=self.start_year,
            end_year=self.end_year,
            input_vars=self.input_vars,
        )
        self.ds_x, self.ds_y = self.fetch_timeseries(
            self.forecast_step
        )  # Precompute pairs
        self.length = self.ds_x.sizes["time"]  # Update length based on valid pairs

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

    # or use with data loader for pytorch as follows:
    # pytorch_dataset = PyTorchERA5Dataset(train_dataset)
    # data_loader = DataLoader(pytorch_dataset, batch_size=16, shuffle=True)
    # etc....
