#!/usr/bin/env bash
#SBATCH --account=m3586
#SBATCH --constraint=gpu
#SBATCH --mail-user=kevin.gill@wisc.edu
#SBATCH --mail-type=ALL

#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH --nodes=2
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
import time

import numpy as np

from bes_ml2.train_velo import BES_Trainer
from bes_ml2 import elm_prediction_datamodule
from bes_ml2 import elm_lightning_model

logger_hash = int(os.getenv('UNIQUE_IDENTIFIER', 0))
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

    block_cols   = [1, 3, 5, 7]
    row_stride   = 1
    row_offset   = 4
    R_sel        = len(np.arange(8)[row_offset::row_stride])
    C_sel        = len(block_cols)

    datamodule = elm_prediction_datamodule.ELM_Prediction_Datamodule(
            data_file="/pscratch/sd/k/kevinsg/bes_ml_jobs/confinement_data/elmcollection_Oursv2.hdf5",
            signal_window_size=48,
            target_sampling_hz=1_000_000.0,
            block_cols=block_cols,
            row_stride=row_stride,
            row_offset=row_offset,
            split_method="event",
            fraction_validation=0.15,
            fraction_test=0.15,
            split_train_data_per_gpu=True,
            standardize_signals=False,
            world_size=world_size,
            num_workers=4,
            batch_size=1024,
        )

    weight_decay = 0.00001
    trial_name = f'{logger_hash}'

    lightning_model = elm_lightning_model.Lightning_Model(
        encoder_lr=1e-3,
        decoder_lr=1e-3,
        signal_window_size=datamodule.signal_window_size,
        n_rows=R_sel,
        n_cols=C_sel,
        monitor_metric='sum_loss/val',
        lr_scheduler_threshold=1e-3,
        lr_scheduler_patience=5,
        weight_decay=weight_decay,
        encoder_type='none',
        mlp_layers=(50, 50),
        mlp_dropout=0.001,
        elm_prediction_mlp=True,
    )

    trainer = BES_Trainer(
        lightning_model=lightning_model,
        datamodule=datamodule,
        experiment_dir='./exp_gill01',
        trial_name=trial_name,
        wandb_log=True,
        log_freq=100,
    )

    trainer.run_all(
        max_epochs=2,
        early_stopping_min_delta=2e-3,
        early_stopping_patience=100,
        skip_test=False,
        skip_predict=False,
        float_precision=32,
    )
    print(f'Python elapsed time {(time.time()-t_start)/60:.1f} min')
except:
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