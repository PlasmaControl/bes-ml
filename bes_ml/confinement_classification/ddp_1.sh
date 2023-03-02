#!/usr/bin/env bash

#SBATCH --account=m3586
#SBATCH --constraint=gpu
#SBATCH --time=00:30:00
#SBATCH --qos=debug

#SBATCH --array=0-0%8

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH --mail-user=kevin.gill@wisc.edu
#SBATCH --mail-type=ALL

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

# NODES_PER_TRAINER=2
# TASKS_PER_TRAINER=$(( NODES_PER_TRAINER * SLURM_NTASKS_PER_NODE ))
# NUMBER_TRAINERS=$(( SLURM_NNODES / NODES_PER_TRAINER ))

# echo Nodes per trainer: $NODES_PER_TRAINER
# echo Tasks per trainer: $TASKS_PER_TRAINER
# echo Number of trainers: $NUMBER_TRAINERS

# export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"

# export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=INFO  
# export NCCL_DEBUG_SUBSYS=ALL

export MASTER_ADDR=$SLURMD_NODENAME
export MASTER_PORT=29500
if [[ -n $SLURM_ARRAY_JOB_ID ]]; then
    export UNIQUE_IDENTIFIER=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
else
    export UNIQUE_IDENTIFIER=$SLURM_JOBID
fi
# export INIT_METHOD_FILE=file://${SLURM_SUBMIT_DIR}/init_file_${UNIQUE_IDENTIFIER}.tmp

echo MASTER_ADDR: $MASTER_ADDR
echo MASTER_PORT: $MASTER_PORT
echo UNIQUE_IDENTIFIER: $UNIQUE_IDENTIFIER
echo INIT_METHOD_FILE: $INIT_METHOD_FILE

JOB_DIR=${HOME}/ml/scratch/job_$UNIQUE_IDENTIFIER
mkdir $JOB_DIR || exit
cd $JOB_DIR || exit
echo Job directory: $PWD

START_TIME=$(date +%s)
srun -u python ${SLURM_SUBMIT_DIR}/ddp_train_0.py
EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
echo Elapsed time $ELAPSED s

exit $EXIT_CODE