#!/bin/bash

zarr_store="/glade/derecho/scratch/katelynw/era5/rechunked_stacked_uncompressed_test.zarr"

bash time_throughput.sh ${zarr_store}
bash parse_throughput.sh timings.txt > summary.txt
