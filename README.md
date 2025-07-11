# Accelerating AI/ML Workflows in Earth Sciences with GPU-Native Xarray and Zarr

🏔️⚡ A collaborative benchmarking and optimization effort from NSF-NCAR, Development Seed, and NVIDIA to accelerate data-intensive geoscience AI/ML workflows using GPU-native technologies like Zarr v3, CuPy, KvikIO, and NVIDIA DALI.

## 📌 Overview

This repository contains code, benchmarks, and examples from Xarray on GPUs hackathon project during the
[NREL/NCAR/NOAA Open Hackathon](https://www.openhackathons.org/s/siteevent/a0CUP00000rwYYZ2A2/se000355)
in Golden, Colorado from 18-27 February 2025. The goal of this project is to provide a proof-of-concept example of optimizing the performance of geospatial machine learning workflows on GPUs by using [Zarr v3]() and [NVIDIA DALI](). 

📖 [Read the full blog post](www.xarray.dev)

In this project, we demonstrate how to:

- Optimize chunking strategies for Zarr datasets
- Read ERA5 Zarr v3 data directly into GPU memory using CuPy and KvikIO
- Apply GPU-based decompression using NVIDIA's nvCOMP
- Build end-to-end GPU-native DALI pipelines
- Improve training throughput for U-Net-based ML models


## 📂 Repository Structure

In this repository, you will find the following:

- `benchmarks/`: Scripts to evaluate read and write performance for Zarr v3 datasets on both CPU and GPU.
- `zarr_ML_optimization`: Contains an example benchmark for training a U-Net model using DALI with Zarr data format.
- 

- ``: Contains the code for the end-to-end example of training a UNet model using the DALI library with Zarr data format, a bunch of command line arguments are added to the script to make it easier to use for benchmarking.


# Getting started

## Basic

Start by cloning the repo & setting up the `conda` environment:
```bash
git clone https://github.com/pangeo-data/ncar-hackathon-xarray-on-gpus.git
cd ncar-hackathon-xarray-on-gpus
conda env create --file environment.yml
conda activate gpuhackathon
```

### Advanced using `conda-lock`

This is for those who want full reproducibility of the virtual environment.
Create a virtual environment with just Python and conda-lock installed first.

```
    mamba create --name gpuhackathon python=3.11 conda-lock=2.5.7
    mamba activate gpuhackathon
```

Generate a unified [`conda-lock.yml`](https://github.com/conda/conda-lock) file
based on the dependency specification in `environment.yml`. Use only when
creating a new `conda-lock.yml` file or refreshing an existing one.
```
    conda-lock lock --mamba --file environment.yml --platform linux-64 --with-cuda=12.8
```

Installing/Updating a virtual environment from a lockile. Use this to sync your
dependencies to the exact versions in the `conda-lock.yml` file.

```
    conda-lock install --mamba --name gpuhackathon conda-lock.yml
```
See also https://conda.github.io/conda-lock/output/#unified-lockfile for more
usage details.


### Running the Examples

For running the benchmark examples and 





