import os
import logging
import socket
import numpy as np
import torch

logger = logging.getLogger(__name__)

def setup_logging(world_rank: int, level: int = logging.INFO) -> None:
    """Sets up basic logging. Logs only from rank 0."""
    if world_rank == 0:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] : %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        logging.basicConfig(level=logging.CRITICAL + 1)

def set_random_seeds(random_seed=0):
    """
    Sets random seeds for reproducibility
    """
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(random_seed)

def init_process_group(
    distributed: bool, backend: str = "nccl"
) -> tuple[int, int, int]:
    """
    Initialize the process group for distributed training.
    """
    if distributed:
        # Try MPI detection first
        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            shmem_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
            
            local_rank = shmem_comm.Get_rank()
            world_size = comm.Get_size()
            world_rank = comm.Get_rank()

            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = comm.bcast(socket.gethostbyname(socket.gethostname()), root=0)

            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = str(np.random.randint(1000, 8000))

        except:
            if "LOCAL_RANK" in os.environ:
                # Environment variables set by torch.distributed.launch or torchrun
                LOCAL_RANK = int(os.environ["LOCAL_RANK"])
                WORLD_SIZE = int(os.environ["WORLD_SIZE"])
                WORLD_RANK = int(os.environ["RANK"])
            elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
                # Environment variables set by mpirun
                LOCAL_RANK = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
                WORLD_SIZE = int(os.environ["OMPI_COMM_WORLD_SIZE"])
                WORLD_RANK = int(os.environ["OMPI_COMM_WORLD_RANK"])
            elif "PMI_RANK" in os.environ:
                # Environment variables set by cray-mpich
                LOCAL_RANK = int(os.environ["PMI_LOCAL_RANK"])
                WORLD_SIZE = int(os.environ["PMI_SIZE"])
                WORLD_RANK = int(os.environ["PMI_RANK"])
            else:
                raise RuntimeError(
                    "Can't find the environment variables for local rank!"
                )
    else: # Non-distributed mode
        # for running without torchrun or mpirun (i.e. ./train_unet.py)
        LOCAL_RANK = 0
        WORLD_SIZE = 1
        WORLD_RANK = 0
    
    # ---------------------
    # Initialize distributed training
    if distributed:
        torch.distributed.init_process_group(
            backend=backend, rank=WORLD_RANK, world_size=WORLD_SIZE
        )
        torch.cuda.set_device(LOCAL_RANK)
    if WORLD_RANK == 0:
        print("----Distrbuted Setup-----")
        print("LOCAL_RANK  : ", LOCAL_RANK)
        print("WORLD_SIZE  : ", WORLD_SIZE)
        print("WORLD_RANK  : ", WORLD_RANK)
        print("cuda device : ", torch.cuda.device_count())
        print("pytorch version : ", torch.__version__)
        print("nccl version : ", torch.cuda.nccl.version())
        print("torch config : ", torch.__config__.show())
        print(torch.__config__.parallel_info())
        print("-------------------------")

    return LOCAL_RANK, WORLD_SIZE, WORLD_RANK