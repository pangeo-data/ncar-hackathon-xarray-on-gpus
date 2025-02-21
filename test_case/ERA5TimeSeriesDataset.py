#!/usr/bin/env python3
"""
This file contains two classes:
1. ERA5TimeSeriesDataset: A custom dataset class to load multiple years of ERA5 data. (No PyTorch dependency)
2. PyTorchERA5Dataset: A wrapper class to use the custom dataset in PyTorch DataLoader.
"""
import os
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader


# -------------------------------------------------------------------------
# 1. ERA5TimeSeriesDataset Class
# -------------------------------------------------------------------------
class ERA5TimeSeriesDataset:
    """
    Load multiple years of ERA5 and forcing datasets from Zarr. 
    Each __getitem__(index) returns (input, target) as NumPy arrays.
    """

    def __init__(self, data_path, year_start, year_end, input_vars, target_vars=None, time_dim='time'):
        """
        Initializes the dataset.

        Args:
            data_path (str): Path to the zarr store.
            year_start (int): Start year for the dataset.
            year_end (int): End year for the dataset.
            input_vars (list): List of input variable names.
            target_vars (list, optional): List of target variable names. Defaults to input_vars.
            time_dim (str): Name of the time dimension. Defaults to 'time'.
        """
        self.data_path = data_path
        self.year_start = year_start
        self.year_end = year_end
        self.input_vars = input_vars
        self.target_vars = target_vars if target_vars is not None else input_vars
        self.time_dim = time_dim

        # load all zarr:
        self.all_files = self._load_data()
        self.length = sum(ds.sizes[self.time_dim] - 1 for ds in self.all_files.values())

    def _load_data(self):
        """Loads all zarr files into a dictionary keyed by year."""
        data_dict = {}
        for year in range(self.year_start, self.year_end + 1):
            zarr_path = os.path.join(self.data_path, f"SixHourly_y_TOTAL_{year}-01-01_{year}-12-31_staged.zarr")
            if os.path.exists(zarr_path):
                ds = xr.open_zarr(zarr_path, consolidated=True)
                # Keep only input variables (note: same set used for target if not specified otherwise)
                data_dict[year] = ds[self.input_vars]
            else:
                print(f"Data for year {year} not found!!!")
        return data_dict

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.length

    def __getitem__(self, index):
        """
        Fetches data at the given index, handling cross-year indexing.
        Each sample consists of the current atmosphere state (input) and 
        the next time step atmosphere state (target).

        Args:
            index (int): Global index for the dataset.
        Returns:
            tuple: (x_data, y_data) as NumPy arrays of shape (vars, lat, lon).
        """
        year, index_in_year = self._find_year_index(index)
        ds = self.all_files[year]

        # Current time step
        x_data = ds.isel({self.time_dim: index_in_year}).to_array().values  # shape: (vars, lat, lon)

        # Next time step (could be next index in the same year or the 0th index in the next year)
        if index_in_year + 1 < ds.sizes[self.time_dim]:
            y_data = ds.isel({self.time_dim: index_in_year + 1}).to_array().values
        else:
            next_year = year + 1
            if next_year in self.all_files:
                y_data = self.all_files[next_year].isel({self.time_dim: 0}).to_array().values
            else:
                raise IndexError("No next time step available for the last entry in the dataset.")

        return x_data, y_data

    def _find_year_index(self, index):
        """
        Determines the corresponding year and local index within that year.

        Args:
            index (int): Global dataset index.
        Returns:
            tuple: (year, local index within the year's dataset)
        """
        accumulated = 0
        for year, ds in self.all_files.items():
            year_length = ds.sizes[self.time_dim] - 1  # -1 because we need a next time step
            if index < accumulated + year_length:
                return year, index - accumulated
            accumulated += year_length
        raise IndexError(
            f"Index out of range. Dataset length: {self.length}, Requested index: {index}"
        )

    def __repr__(self):
        """Returns a summary of all datasets loaded."""
        summary = [f"ERA5TimeSeriesDataset({self.year_start}-{self.year_end})"]
        for year, ds in self.all_files.items():
            summary.append(f"Year: {year}, Variables: {list(ds.data_vars.keys())}, Shape: {ds.sizes}")
        return "\n".join(summary)


# -------------------------------------------------------------------------
# 2. PyTorch Wrapper Dataset
# -------------------------------------------------------------------------
class PyTorchERA5Dataset(Dataset):
    """
    Wraps the ERA5TimeSeriesDataset so it can be used in PyTorch DataLoader.
    """
    def __init__(self, era5_dataset):
        """
        era5_dataset (ERA5TimeSeriesDataset): An instance of the custom ERA5 dataset.
        """
        self.era5_dataset = era5_dataset

    def __len__(self):
        return len(self.era5_dataset)

    def __getitem__(self, idx):
        x_data, y_data = self.era5_dataset[idx]  # NumPy arrays: (vars, lat, lon)
        x_tensor = torch.from_numpy(x_data).float()  # Convert to float32
        y_tensor = torch.from_numpy(y_data).float()  # Convert to float32
        return x_tensor, y_tensor



        


if __name__ == "__main__":

    ## Example usage of the ERA5TimeSeriesDataset class
    data_path = "/glade/derecho/scratch/ksha/CREDIT_data/ERA5_mlevel_arXiv"
    year_start = 1990
    year_end = 2010
    input_vars = ['t2m', 'V500', 'U500', 'T500', 'Z500', 'Q500']

    dataset = ERA5TimeSeriesDataset(data_path, year_start, year_end, input_vars=input_vars)

    print(dataset)

    index = 1000 
    x, y = dataset[index]

    print("Input (current atmosphere state):", x.shape)
    print("Target (next time step atmosphere state):", y.shape)
