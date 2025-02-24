#!/usr/bin/env python3
from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
#from nvidia.dali.plugin.pytorch import DALIGenericIterator

from test_case.ERA5TimeSeriesDataset import ERA5Dataset, PyTorchERA5Dataset


@pipeline_def
def create_dali_pipeline(dataset, batch_size, device='gpu'):
    """
    Creates a DALI pipeline for loading and preprocessing ERA5 time-series data.

    Args:
        dataset (ERA5TimeSeriesDataset): The dataset to load from.
        batch_size (int): Batch size for the pipeline.
        device (str): Device to use ('gpu' or 'cpu'). Defaults to 'gpu'.
    """
    # Define external source to fetch data from the dataset
    inputs, targets = fn.external_source(
        source=dataset,
        num_outputs=2,
        dtype=types.FLOAT,
        device=device,
        parallel=True,
        batch=False,
    )

    # Normalize inputs and targets (example preprocessing step)
    inputs = fn.normalize(inputs, mean=0.0, stddev=1.0)
    targets = fn.normalize(targets, mean=0.0, stddev=1.0)

    # Resize inputs and targets to a fixed size (example preprocessing step)
    inputs = fn.resize(inputs, resize_x=256, resize_y=256)
    targets = fn.resize(targets, resize_x=256, resize_y=256)

    return inputs, targets


# Example usage:
if __name__ == "__main__":
    ## Example usage of the ERA5TimeSeriesDataset class
    data_path = "/glade/derecho/scratch/ksha/CREDIT_data/ERA5_mlevel_arXiv"
    start_year = 2000
    end_year = 2010

    # for now, just surface variables!cd
    input_vars = ['t2m', 'V500', 'U500', 'T500', 'Z500', 'Q500']
    target_vars = ['t2m']


    #dataset = ERA5TimeSeriesDataset(data_path, start_year, end_year, input_vars=input_vars)

    era5_dataset = ERA5Dataset(
        data_path=data_path,
        start_year=start_year,
        end_year=end_year,
        input_vars=input_vars,
        target_vars=target_vars
    )

    
    print(era5_dataset)       
    print(f"Total samples: {len(era5_dataset)}")

    # Create DALI pipeline
    batch_size = 32
    pipe = create_dali_pipeline(era5_dataset, batch_size=batch_size, device='gpu', num_threads=4, device_id=0)
    pipe.build()

    # Create DALI iterator
    dali_iter = DALIGenericIterator(pipe, output_map=["inputs", "targets"], size=len(era5_dataset))

    # Fetch a batch of data
    for batch in dali_iter:
        inputs = batch[0]["inputs"]
        targets = batch[0]["targets"]
        print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
        break # just a test
