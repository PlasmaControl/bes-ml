import os
import socket
from pathlib import Path
from datetime import timedelta

import torch
import torch.distributed

import dataclasses

try:
    from ..base.train_base import Trainer_Base
    from .confinement_data_v2 import Confinement_Data_v2
except ImportError:
    from bes_ml.base.train_base import Trainer_Base
    from bes_ml.confinement_classification.confinement_data_v2 import Confinement_Data_v2

from bes_ml.elm_regression import Trainer

WORLD_SIZE = int(os.environ.get('SLURM_NTASKS'))
WORLD_RANK = int(os.environ.get('SLURM_PROCID'))
LOCAL_RANK = int(os.environ.get('SLURM_LOCALID'))
UNIQUE_IDENTIFIER = os.environ.get('UNIQUE_IDENTIFIER')    

@dataclasses.dataclass(eq=False)
class Trainer(
    Confinement_Data_v2,  # confinement data
    Trainer_Base,  # training and output
):

    def __post_init__(self) -> None:

        self.mlp_output_size = 4
        self.is_classification = True
        self.is_regression = not self.is_classification
        self.is_ddp = True
        self.world_size = int(os.environ.get('SLURM_NTASKS'))
        self.world_rank = int(os.environ.get('SLURM_PROCID'))
        self.local_rank = int(os.environ.get('SLURM_LOCALID'))
        UNIQUE_IDENTIFIER = os.environ.get('UNIQUE_IDENTIFIER')    
        super().__post_init__()  # Trainer_Base.__post_init__()

if __name__=='__main__':
    # print(
    #     f"Host {socket.gethostname()} "
    #     # f"MASTER_ADDR {os.environ['MASTER_ADDR']} "
    #     # f"MASTER_PORT {os.environ['MASTER_PORT']} "
    #     f"NNODES {os.environ['SLURM_NNODES']} "
    #     f"NTASKS {os.environ['SLURM_NTASKS']} "
    #     f"PROCID {os.environ['SLURM_PROCID']} "
    #     f"LOCALID {os.environ['SLURM_LOCALID']} "
    # )

    # MASTER_ADDR = os.environ.get('MASTER_ADDR')
    # MASTER_PORT = os.environ.get('MASTER_PORT')
    # WORLD_SIZE = int(os.environ.get('SLURM_NTASKS'))
    # WORLD_RANK = int(os.environ.get('SLURM_PROCID'))
    # LOCAL_RANK = int(os.environ.get('SLURM_LOCALID'))
    # UNIQUE_IDENTIFIER = os.environ.get('UNIQUE_IDENTIFIER')    
    # print(
    #     f"WORLD_SIZE {WORLD_SIZE} "
    #     f"WORLD_RANK {WORLD_RANK} "
    #     f"LOCAL_RANK {LOCAL_RANK} "
    # )
    # assert LOCAL_RANK <= WORLD_RANK
    # assert WORLD_RANK < WORLD_SIZE

    # master_node=os.environ.get('SLURMD_NODENAME')
    # init_method_tcp = f'tcp://{master_node}:29500'
    # print(f'init_method_tcp: {init_method_tcp}')

    # torch.distributed.init_process_group(
    #     backend='nccl' if torch.cuda.is_available() else 'gloo',
    #     world_size=WORLD_SIZE,
    #     rank=WORLD_RANK,
    #     timeout=timedelta(seconds=10),
    #     # init_method=init_method_tcp,
    #     # init_method=os.environ.get('INIT_METHOD_FILE'),
    # )

    Trainer(
        dense_num_kernels=8,
        fraction_test=0.1,
        dataset_to_ram=True,
        do_train=True,
    )