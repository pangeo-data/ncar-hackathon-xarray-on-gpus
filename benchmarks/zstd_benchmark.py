"""
Zarr v3 I/O Benchmark: CPU vs GPU Read Performance

This script benchmarks the I/O performance of writing and reading a synthetic
Zarr v3 dataset using CPU and GPU. It demonstrates how to:

- Create a 4D array in Zarr v3 using a specified compression codec (CPU or GPU).
- Read the dataset using either CPU-based or GPU-accelerated access.
- Annotate profiling regions using NVTX for use with NVIDIA Nsight tools.
- Compute and report effective I/O bandwidth in GB/s.

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


def get_store(path: Path) -> LocalStore:
    async def _get_store(path: Path) -> LocalStore:
        return await LocalStore.open(path)

    return asyncio.run(_get_store(path))

@nvtx.annotate(color="red", domain="benchmark")
def write(
    size: tuple[int, int, int, int],
    chunks: tuple[int, int, int, int],
    store: Store,
    write_codec: str | Codec,
    read_codec: str | Codec,
) -> None:
    src = np.random.uniform(size=size).astype(np.float32)  # allocate on CPU
    z = zarr.create_array(
        store,
        name="a",
        shape=src.shape,
        chunks=chunks,
        dtype=src.dtype,
        overwrite=True,
        zarr_format=3,
        compressors=write_codec,
    )
    z[:] = src

@nvtx.annotate(color="red", domain="benchmark")
def read(
    size: tuple[int, int, int, int],
    store: Store,
    gpu: bool = True,
) -> None:
    with zarr.config.enable_gpu() if gpu else nullcontext():
        g = zarr.open_group(store=store)
        a = g.get("a")
        print(f"Opened array with compressors: {a.compressors}")

        color = "green" if gpu else "blue"
        start_time = time.time()
        for i in range(size[0] // 2):
            with nvtx.annotate(message=f"iteration {i}", color=color):
                start_time_index = 2 * i
                end_time_index = 2 * (i + 1)
                result = a[start_time_index:end_time_index, :, :, :]
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_bytes_gb = 2.0 * (size[0] // 2) * math.prod(size[1:]) / (1024.0) ** 3
        print(f"Total time to read data: {elapsed_time} s")
        print(f"Effective I/O bandwidth: {total_bytes_gb / elapsed_time} GB/s")


if __name__ == "__main__":
    cpu_codec = ZstdCodec()
    gpu_codec = NvcompZstdCodec()

    dataset_size = (10, 128, 1280, 640)
    chunks = (1, 1, 1280, 640)

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "benchmark.zarr"
        store = get_store(path)
        write(dataset_size, chunks, store, cpu_codec, gpu_codec)
        read(dataset_size, store, gpu=False)
        read(dataset_size, store, gpu=True)
