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
import time

import numpy as np

from bes_ml2.train_velo import BES_Trainer
from bes_ml2 import velocimetry_datamodule
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

    n_rows = 7
    n_cols = 8

    datamodule = velocimetry_datamodule.Velocimetry_Datamodule(
            data_file='/pscratch/sd/k/kevinsg/bes_ml_jobs/confinement_data/20250111_vZ_maxlag150_bandpass_turb.hdf5',
            signal_window_size=100,
            n_rows=n_rows,
            n_cols=n_cols,
            batch_size=256,
            num_workers=4,
            seed=0,
            world_size=world_size,
            lower_cutoff_frequency_hz=60e3,
            upper_cutoff_frequency_hz=250e3,  # Upper cutoff frequency in Hz
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
            train_shots=['145384', '145385', '145388'],
            validation_shots=['145391'],
            test_shots=['145387'],
            predict_shots=['145384', '145387'],
            # train_shots=['199779', '191973', '191715', '193108', '191714', '199782', '191971', '200355', '200063', '199754', '191975', '191676', '193253', '191972', '199775', '191670', '193257', '199748', '200024', '193112', '193248', '200062', '200354', '199753', '199755', '193110', '200021', '193107', '191976', '193113', '193111', '191674', '200349'],
            # validation_shots=[ '199749',  '191710', '199795', '191965'],
            # test_shots=['199718', '191673', '191757', '199756'],
            # predict_shots=['191670', '199718', '191714'],
            # train_shots=['191670', '191673', '191674', '191676', '191714', '191754'],
            # validation_shots=['191670'],
            # test_shots=['191670'],
            # predict_shots=['191670', '191714'],
            split_train_data_per_gpu=True,
            vZ_uncertainty_threshold=5.0,
            # target_labels=["smoothed_vZ_window25", "smoothed_vZ_uncertainty_window25"],
            target_labels=["vZ", "vZ_uncertainty"],
            do_flip_augmentation=True,
        )

    weight_decay = 0.001
    trial_name = f'{logger_hash}'

    lightning_model = elm_lightning_model.Lightning_Model(
        encoder_lr=1e-3,
        decoder_lr=1e-3,
        signal_window_size=datamodule.signal_window_size,
        n_rows=n_rows,
        n_cols=n_cols,
        monitor_metric='sum_loss/val',
        lr_scheduler_threshold=1e-3,
        lr_scheduler_patience=10,
        weight_decay=weight_decay,
        encoder_type='none',
        cnn_nlayers=2,
        cnn_num_kernels=[16, 32],
        cnn_kernel_time_size=[8, 4],
        cnn_kernel_spatial_size=[3, 3],
        cnn_padding = [1, 1],
        cnn_maxpool_spatial_size = [2, 1],
        cnn_maxpool_time_size = [2, 2],
        leaky_relu_slope=0.001,
        mlp_layers=(512, 256, 128),
        # encoder_type='rcn',
        # rcn_reservoir_size=1000,
        # rcn_spectral_radius=0.9,
        # rcn_sparsity=0.1,
        # rcn_input_scaling=1.0,
        # rcn_leaky_rate=0.5,            
        # mlp_layers=(512, 256, 128, 64),
        mlp_dropout=0.1,
        velocimetry_mlp=True,
        save_test_data=False,
    )

    trainer = BES_Trainer(
        lightning_model=lightning_model,
        datamodule=datamodule,
        experiment_dir='./exp_gill01',
        trial_name=trial_name,
        wandb_log=True,
        log_freq=1,
        # num_val_batches=1,
        # num_train_batches=100, 
        # val_check_interval=500,
        # num_train_batches=1000, # for 4 nodes full dataset
        val_check_interval=5000,
        num_train_batches=19900,
    )

    trainer.run_all(
        max_epochs=5,
        early_stopping_min_delta=2e-3,
        early_stopping_patience=10,
        skip_test=False,
        skip_predict=False,
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