#!/usr/bin/env bash
#SBATCH --account=m3586
#SBATCH --constraint=gpu
#SBATCH --mail-user=kevin.gill@wisc.edu
#SBATCH --mail-type=ALL

#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=3

#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --qos=debug
###SBATCH --array=0

echo Python executable: $(which python)
echo
echo Job name: $SLURM_JOB_NAME
echo QOS: $SLURM_JOB_QOS
echo Account: $SLURM_JOB_ACCOUNT
echo Submit dir: $SLURM_SUBMIT_DIR
echo
echo Job array ID: $SLURM_ARRAY_JOB_ID
echo Job ID: $SLURM_JOBID
echo Job array task: $SLURM_ARRAY_TASK_ID
echo Job array task count: $SLURM_ARRAY_TASK_COUNT
echo
echo Nodes: $SLURM_NNODES
echo Head node: $SLURMD_NODENAME
echo hostname $(hostname)
echo Nodelist: $SLURM_NODELIST
echo Tasks per node: $SLURM_NTASKS_PER_NODE
echo GPUs per node: $SLURM_GPUS_PER_NODE

if [[ -n $SLURM_ARRAY_JOB_ID ]]; then
    export UNIQUE_IDENTIFIER=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
else
    export UNIQUE_IDENTIFIER=$SLURM_JOBID
fi
echo UNIQUE_IDENTIFIER: $UNIQUE_IDENTIFIER

JOB_DIR=/pscratch/sd/k/kevinsg/bes_ml_jobs/
mkdir --parents $JOB_DIR || exit
cd $JOB_DIR || exit
echo Job directory: $PWD

export WANDB__SERVICE_WAIT=300

PYTHON_SCRIPT=$(cat << END

import sys
import os
from pathlib import Path
import time
from datetime import datetime, timedelta

import numpy as np

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy
import torch
import wandb

from bes_ml2.train_velo import BES_Trainer
from bes_ml2 import velocimetry_datamodule
from bes_ml2 import elm_lightning_model

logger_hash = int(os.getenv('UNIQUE_IDENTIFIER', 0))
num_nodes = int(os.getenv('SLURM_NNODES', '1'))  # Default to 1 if not set
world_size = int(os.getenv('SLURM_NTASKS', 0))
world_rank = int(os.getenv('SLURM_PROCID', 0))
local_rank = int(os.getenv('SLURM_LOCALID', 0))
node_rank = int(os.getenv('SLURM_NODEID', 0))
print(f'World rank {world_rank} of {world_size} (local rank {local_rank} on node {node_rank})')

is_global_zero = world_rank == 0

if not is_global_zero:
    f = open(os.devnull, 'w')
    sys.stdout = f

try:
    t_start = time.time()

    trial_name = f'{logger_hash}'
    experiment_dir = './exp_gill01'
    wandb_log = True

    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = experiment_dir.name
    experiment_parent_dir = experiment_dir.parent

    # set loggers
    tb_logger = TensorBoardLogger(
        save_dir=experiment_parent_dir,
        name=experiment_name,
        version=trial_name,
        default_hp_metric=False,
    )
    trial_dir = Path(tb_logger.log_dir).absolute()
    print(f"Trial directory: {trial_dir}")
    loggers = [tb_logger]

    checkpoint = Path('/pscratch/sd/k/kevinsg/bes_ml_jobs/exp_gill01/34849261_copy/checkpoints/last.ckpt') 

    # load data and model from checkpoint
    lightning_model = elm_lightning_model.Lightning_Model.load_from_checkpoint(checkpoint_path=checkpoint)
    # datamodule = velocimetry_datamodule.Velocimetry_Datamodule.load_from_checkpoint(checkpoint_path=checkpoint)
    datamodule = velocimetry_datamodule.Velocimetry_Datamodule(
        data_file='/pscratch/sd/k/kevinsg/bes_ml_jobs/confinement_data/20241207_vZ_smooth.hdf5',
        signal_window_size=100,
        n_rows=8,
        n_cols=7,
        batch_size=256,
        num_workers=4,
        seed=0,
        world_size=world_size,
        lower_cutoff_frequency_hz=50e3,
        upper_cutoff_frequency_hz=120e3,  # Upper cutoff frequency in Hz
        signal_mean=0.007,
        signal_stdev=0.060,
        start_time_ms=2400,
        standardize_labels=False,
        clip_labels=False,
        labels_lower_bound=-151.0,
        labels_upper_bound=151.0,
        label_mean=-3.22,
        label_std=14.40,
        # normalize_labels=True,   
        label_min=-50,           
        label_max=50,       
        split_method='shot',
        fraction_validation=0.1,
        fraction_test=0.05,
        train_shots=['191754', '191676', '191673'],
        validation_shots=['191676'],
        test_shots=['191676'],
        predict_shots=['191714', '191670'],
        split_train_data_per_gpu=True,
        vZ_uncertainty_threshold=10.0,
        # target_labels=["smoothed_vZ_window25", "smoothed_vZ_uncertainty_window25"],
        target_labels=["vZ", "vZ_uncertainty"],
        do_flip_augmentation=True,
    )
    datamodule.setup(stage='predict')

    if wandb_log:
        wandb.login()
        wandb_logger = WandbLogger(
            save_dir=experiment_dir,
            project=experiment_name,
            name=trial_name,
        )
        wandb_logger.watch(
            lightning_model, 
            log='all', 
            log_freq=100,
        )
        loggers.append(wandb_logger)

    float_precision = '16-mixed' if torch.cuda.is_available() else 32
    frontends_active = [value for value in lightning_model.frontends_active.values()]
    some_unused = False in frontends_active

    # Initialize the Trainer
    trainer = Trainer(
        logger=loggers,
        num_nodes=num_nodes,
        precision=float_precision,
        strategy=DDPStrategy(find_unused_parameters=some_unused, timeout=timedelta(seconds=9600)),
    )

    # trainer.test(model=lightning_model, datamodule=datamodule)
    trainer.predict(model=lightning_model, datamodule=datamodule, ckpt_path='last')

    print(f'Python elapsed time {(time.time()-t_start)/60:.1f} min')
except Exception as e:
    print(f"An error occurred: {e}")
    if not is_global_zero:
        f.close()
        sys.stdout = sys.__stdout__
    raise
finally:
    if not is_global_zero:
        f.close()
        sys.stdout = sys.__stdout__
END
)

echo Script:
echo "${PYTHON_SCRIPT}"


START_TIME=$(date +%s)
srun python -c "${PYTHON_SCRIPT}"
EXIT_CODE=$?
END_TIME=$(date +%s)
echo Slurm elapsed time $(( (END_TIME - START_TIME)/60 )) min $(( (END_TIME - START_TIME)%60 )) s

exit $EXIT_CODE