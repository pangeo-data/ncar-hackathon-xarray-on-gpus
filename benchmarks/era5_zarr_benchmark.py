"""
This script is a GPU/CPU I/O benchmark for reading a Zarr dataset.
It compares read performance between CPU-based and GPU-native approaches using Zarr v3.

The script uses:
- `zarr.config.enable_gpu()` for GPU-backed reads (via CuPy),
- `GDSStore` from `kvikio_zarr_v3` for GPU Direct Storage (GDS) support,
- `nvtx` annotations for profiling iterations with NVIDIA Nsight tools.

The dataset is assumed to be a 4D array stored under the key 'combined', typically in (time, channel, height, width) format.

The benchmark:
- Reads pairs of time steps in a loop,
- Measures elapsed time,
- Computes effective I/O bandwidth in GB/s.
"""
import asyncio
from contextlib import nullcontext
import math
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import nvtx

import zarr
from zarr.abc.codec import Codec
from zarr.abc.store import Store
from zarr.codecs import NvcompZstdCodec, ZstdCodec
from zarr.storage import LocalStore

from kvikio_zarr_v3 import GDSStore


def get_store(path: Path, cls: Store = LocalStore) -> LocalStore:
    async def _get_store(path: Path) -> LocalStore:
        return await cls.open(path)

    return asyncio.run(_get_store(path))

@nvtx.annotate(color="red", domain="benchmark")
def read(
    store: Store,
    gpu: bool = True,
) -> None:
    with zarr.config.enable_gpu() if gpu else nullcontext():
        g = zarr.open_group(store=store)
        a = g.get("combined")
        size = tuple(a.shape)
        print(f"Opened array with compressors: {a.compressors}")

        color = "green" if gpu else "blue"
        start_time = time.time()
        niters = min(100, size[0] // 2)
        for i in range(niters):
            with nvtx.annotate(message=f"iteration {i}", color=color):
                start_time_index = 2 * i
                end_time_index = 2 * (i + 1)
                result = a[start_time_index:end_time_index, :, :, :]
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_bytes_gb = 2.0 * (niters) * math.prod(size[1:]) / (1024.0) ** 3
        print(f"Total time to read data: {elapsed_time} s")
        print(f"Effective I/O bandwidth: {total_bytes_gb / elapsed_time} GB/s")


if __name__ == "__main__":
    path = Path("/glade/derecho/scratch/katelynw/era5/rechunked_stacked_test.zarr")
    store = get_store(path, cls=GDSStore)
    read(store, gpu=False)
    read(store, gpu=True)
