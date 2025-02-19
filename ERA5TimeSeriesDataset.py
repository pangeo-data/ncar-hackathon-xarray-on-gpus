#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import os
import xarray as xr
from torch.utils.data import Dataset, DataLoader


class ERA5TimeSeriesDataset(Dataset):
    """
    Load multiple years of ERA5 and forcing datasets from Zarr. 
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
        """Loads all zarr files into a dictionary."""
        data_dict = {}
        for year in range(self.year_start, self.year_end + 1):
            zarr_path = os.path.join(self.data_path, f"SixHourly_y_TOTAL_{year}-01-01_{year}-12-31_staged.zarr")
            if os.path.exists(zarr_path):
                data_dict[year] = xr.open_zarr(zarr_path, consolidated=True)
            else: 
                print ("Data for year {} not found!!!".format(year))
        return data_dict

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.length

    def __getitem__(self, index):
        """
        Fetches data at the given index, handling cross-year indexing.

        Args:
            index (int): Index to fetch.
        Returns:
            tuple: (input tensor, target tensor)
        """
        year, index_in_year = self._find_year_index(index)
        ds = self.all_files[year]
        
        x_data = ds.isel({self.time_dim: index_in_year})[self.input_vars].to_array().values
        y_data = ds.isel({self.time_dim: index_in_year + 1})[self.target_vars].to_array().values

        #x = torch.tensor(x_data, dtype=torch.float32).view(-1)
        #y = torch.tensor(y_data, dtype=torch.float32).view(-1)
        
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
            year_length = ds.sizes[self.time_dim] - 1
            if index < accumulated + year_length:
                return year, index - accumulated
            accumulated += year_length
        raise IndexError(f"Index out of range. Dataset length: {self.length}, Requested index: {index}")

    def print_all_data(self):
        """Prints a summary of all datasets loaded."""
        for year, ds in self.all_files.items():
            print(f"Year: {year}")
            print(ds)


if __name__ == "__main__":
    data_path = "/glade/derecho/scratch/ksha/CREDIT_data/ERA5_mlevel_arXiv"
    year_start = 1990
    year_end = 2010
    input_vars = ['SP', 't2m', 'V500', 'U500', 'T500', 'Z500', 'Q500']

    dataset = ERA5TimeSeriesDataset(data_path, year_start, year_end, input_vars=input_vars)
    dataset.print_all_data()

    index = 5  # Pick any time step
    x, y = dataset[index]  # Get input and target

    print("Input (current atmosphere state):", x.shape)
    print("Target (next time step atmosphere state):", y.shape)