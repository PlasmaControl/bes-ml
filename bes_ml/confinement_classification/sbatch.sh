#!/usr/bin/env bash

#SBATCH --account=m3586
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=00:15:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4

#SBATCH --mail-user=kevin.gill@wisc.edu
#SBATCH --mail-type=ALL

export SLURM_CPU_BIND=cores

cd ${HOME}/ml/scratch || exit
echo $PWD

START_TIME=$(date +%s)
srun python ${SLURM_SUBMIT_DIR}/train_v4.py
EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
echo "Elapsed time $ELAPSED s"

exit $EXIT_CODE