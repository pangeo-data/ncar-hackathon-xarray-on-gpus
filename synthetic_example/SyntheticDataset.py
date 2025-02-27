import os

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

Tensor = torch.Tensor

Tensor = torch.Tensor


class ERA5Dataset:
    """
    Load multiple years of ERA5 and forcing datasets from Zarr.
    Each __getitem__(index) returns (input, target) as NumPy arrays.
    """

    def __init__(
        self,
        data_path,
        start_year,
        end_year,
        input_vars,
        target_vars=None,
        forecast_step=1,
    ):
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
        self.normalized = False
        self.forecast_step = forecast_step

        # load all zarr:
        self.dataset = self._load_data()
        self.ds_x, self.ds_y = self.fetch_timeseries(
            self.forecast_step
        )  # Precompute pairs
        self.length = self.ds_x.sizes["time"]  # Update length based on valid pairs

    def _load_data(self):
        """Loads all zarr files into a dictionary keyed by year."""
        zarr_paths = []
        for year in range(self.start_year, self.end_year + 1):
            zarr_path = os.path.join(
                self.data_path,
                f"SixHourly_y_TOTAL_{year}-01-01_{year}-12-31_staged.zarr",
            )
            if os.path.exists(zarr_path):
                zarr_paths.append(zarr_path)
            else:
                print(f"Data for year {year} not found!!!")
        ds = xr.open_mfdataset(
            zarr_paths, engine="zarr", consolidated=True, combine="by_coords"
        )[self.input_vars]
        self.length = ds.sizes["time"]
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

    def normalize(self, mean_file=None, std_file=None):
        """
        Normalize the dataset using the mean and std files.
        """
        if mean_file is not None and std_file is not None:
            mean = xr.open_dataset(mean_file)
            std = xr.open_dataset(std_file)
        else:
            mean = self.dataset.mean(dim="time")
            std = self.dataset.std(dim="time")
        self.dataset = (self.dataset - mean) / std
        self.normalized = True

    def __repr__(self):
        """Returns a summary of all datasets loaded."""
        return self.dataset.__repr__()

    def __getitem__(self, index):
        """Enable direct indexing"""
        x_data = np.zeros([6, 640, 1280], dtype=np.float32)
        y_data = np.zeros([6, 640, 1280], dtype=np.float32)
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
        self.ds_x, self.ds_y = self.era5_dataset.fetch_timeseries(
            forecast_step=self.forecast_step
        )

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.ds_x.sizes["time"]

    def __getitem__(self, index):
        """
        Returns a single sample (input, target) as PyTorch tensors.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: (input_tensor, target_tensor)
        """
        # Extract data at the given index
        x_tensor = torch.zeros([6, 640, 1280], dtype=torch.float32)
        # TODO: I'm surprised this is [6, 640, 1280] rather than [1, 640, 1280]
        y_tensor = torch.zeros([6, 640, 1280], dtype=torch.float32)
        return x_tensor, y_tensor
