# xarray-on-gpus

Repository for the Xarray on GPUs team during the
[NREL/NCAR/NOAA Open Hackathon](https://www.openhackathons.org/s/siteevent/a0CUP00000rwYYZ2A2/se000355)
in Golden, Colorado from 18-27 February 2025.

# Getting started

## Installation

### Basic

To help out with development, start by cloning this [repo-url](/../../)

    git clone <repo-url>

Then I recommend [using mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
to install the dependencies. A virtual environment will also be created with Python and
[JupyterLab](https://github.com/jupyterlab/jupyterlab) installed.

    cd ncar-hackathon-xarray-on-gpus
    mamba env create --file environment.yml

Activate the virtual environment first.

    mamba activate gpuhackathon

Finally, double-check that the libraries have been installed.

    mamba list

### Advanced

This is for those who want full reproducibility of the virtual environment.
Create a virtual environment with just Python and conda-lock installed first.

    mamba create --name gpuhackathon python=3.11 conda-lock=2.5.7
    mamba activate gpuhackathon

Generate a unified [`conda-lock.yml`](https://github.com/conda/conda-lock) file
based on the dependency specification in `environment.yml`. Use only when
creating a new `conda-lock.yml` file or refreshing an existing one.

    conda-lock lock --mamba --file environment.yml --platform linux-64 --with-cuda=12.8

Installing/Updating a virtual environment from a lockile. Use this to sync your
dependencies to the exact versions in the `conda-lock.yml` file.

    conda-lock install --mamba --name gpuhackathon conda-lock.yml

See also https://conda.github.io/conda-lock/output/#unified-lockfile for more
usage details.
