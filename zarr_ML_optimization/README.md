# Zarr ML end-to-end example

## How to run!
To run on 1 GPU, use the following command:

```bash
module load conda
conda activate /glade/work/weiji/conda-envs/gpuhackathon
```

To run on 1 GPU, use the following command:
```
./train_unet.py
```

To run on single node multi-GPU, use the following command:

```bash
torchrun --nnodes=1 --nproc-per-node=4 train_unet.py --distributed
```

To run with nsys:

```bash
module purge
module load ncarenv/23.09
module reset
module load cuda
```
Check the result of the following matches `/glade/u/apps/common/23.08/spack/opt/spack/cuda/12.2.1/bin/nsys`:

```bash
which nsys
```

Then run with nsys profiling with

```
nsys profile -t nvtx,cuda,osrt --gpu-metrics-device all --force-overwrite=true --output=training_benchmark python train_unet.py
```

To run without data loading:

```
export synthetic="True"; nsys profile -t nvtx,cuda,osrt --gpu-metrics-device all --force-overwrite=true --output=training_benchmark python train_unet.py
```

----------------

To run on multi-node multi-GPU, use the following command (full example [here](https://github.com/negin513/distributed-pytorch-hpc/blob/main/scripts/run_mpi.sh)):

```bash
MASTER_ADDR=$head_node_ip MASTER_PORT=1234 mpiexec -np 8 ./train_unet.py --distributed
```

In the above command, replace `$head_node_ip` with the IP address of the head node.
For example using PBS:
``` bash
# Determine the number of nodes:
nnodes=$(< $PBS_NODEFILE wc -l)
nodes=( $( cat $PBS_NODEFILE ) )
head_node=${nodes[0]}
head_node_ip=$(ssh $head_node hostname -i | awk '{print $1}')
```


# 
Average throughput on 1 GPU (A100-40GB): 92.34 samples/sec, ~1000 seconds per epoch, (~35 minutes)
Average throughput on 4 GPUs (A100-40GB): 800.5 samples/sec, ~500 seconds per epoch , (~ 16 minutes)
