cluster="$1"
dd if='/glade/derecho/scratch/maxjones/tmp/large_array.npy' of=/dev/null > array-throughput-${cluster}.txt 2>&1
