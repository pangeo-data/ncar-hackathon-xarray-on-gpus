"""
dali + Zarr (GPU) example.

This script adapts the GPU example from
https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/external_input.html
to use Zarr for storage.

To run it, you'll currently need to use my fork of zarr-python:

    pip install git+https://github.com/TomAugspurger/zarr-python/@tom/fix/gpu

That should be in zarr `main` soon. You'll also need the data.

```
mkdir -p data/images
cd data/images
curl -O https://docs.nvidia.com/deeplearning/dali/user-guide/docs/_images/examples_general_data_loading_external_input_12_2.png
curl -O curl -O https://docs.nvidia.com/deeplearning/dali/user-guide/docs/_images/examples_general_data_loading_external_input_19_2.png

```

And a `file_list.txt` like

```
examples_general_data_loading_external_input_12_2.png 0
examples_general_data_loading_external_input_19_2.png 1
```

Then run `make_data()` to create the zarr store.
"""


import types
from random import shuffle
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import zarr
import zarr.storage
from PIL import Image


batch_size = 16


# create the data
# Right now, assuming a chunksize of 1 along the dimension being sampled.
# We have some interesting options here w.r.t. the chunksize and shuffling.
# 


def make_data():
    # TODO: figure out the shape here.
    # goes from 4 -> 3 somewhere.
    store = zarr.storage.LocalStore(root="data/example.zarr")
    group = zarr.create_group(store, overwrite=True)

    TOTAL_SAMPLES = 100

    # note: the images from the docs vary in size while Zarr requires
    # uniform chunk sizes. I've truncated the images to 231 x 300

    arr = group.create_array(
        name="images",
        shape=(TOTAL_SAMPLES, 231, 300, 3),
        chunks=(1, 231, 300, 3),
        dtype="uint8",
        overwrite=True,
    )

    labels = group.create_array(
        name="labels",
        shape=(TOTAL_SAMPLES,),
        chunks=(1,),
        dtype="uint8",
        overwrite=True,
    )

    # TODO: use file list
    # assuming you've downloaded these two
    img = Image.open(
        "data/images/examples_general_data_loading_external_input_12_2.png"
    )
    arr[0] = img
    labels[0] = 0
    img = Image.open(
        "data/images/examples_general_data_loading_external_input_19_2.png"
    )
    arr[1] = img
    labels[1] = 1


class ExternalInputIterator:
    def __init__(self, batch_size: int):
        self.root = "data/example.zarr/"
        self.variable = "images"
        self.batch_size = batch_size

        # Does this class get serialized? Is it safe to store
        # references to zarr arrays here?
        # self.images = zarr.open_array(self.root, path=self.variable)
        # self.labels = zarr.open_array(self.root, path="labels")

        self.indices = list(
            range(zarr.open_array(self.root, path=self.variable).shape[0])
        )
        shuffle(self.indices)
        self.i = 0
        self.n = len(self.indices)

    def __iter__(self):
        self.i = 0
        self.n = len(self.indices)
        return self

    def __next__(self):
        batch = []
        labels = []

        arr = zarr.open(self.root, path=self.variable)
        arr_labels = zarr.open(self.root, path="labels")

        for _ in range(self.batch_size):
            batch.append(arr[self.i])
            labels.append(arr_labels[self.i])
            self.i = (self.i + 1) % self.n
        return (batch, labels)


def main():
    make_data()
    print (" Data created!")
    eii = ExternalInputIterator(batch_size)
    zarr.config.enable_gpu()
    pipe = Pipeline(batch_size=batch_size, num_threads=2, device_id=0)
    # note: using the `device="gpu"` variant from https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/external_input.html
    with pipe:
        images, labels = fn.external_source(source=eii, num_outputs=2, dtype=types.UINT8, device="gpu")
        enhance = fn.brightness_contrast(images, contrast=2)
        pipe.set_outputs(enhance, labels)

    pipe.build()
    pipe_out = pipe.run()

    batch_cpu = pipe_out[0].as_cpu()
    labels_cpu = pipe_out[1].as_cpu()

    print(batch_cpu.at(0).shape)
    print(labels_cpu.at(0))


if __name__ == "__main__":
    main()
