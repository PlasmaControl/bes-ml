#!/usr/bin/env bash

#SBATCH -A m3586
#SBATCH -C gpu

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128

#SBATCH --gpus-per-task=1

#SBATCH -q debug
#SBATCH -t 30

export SLURM_CPU_BIND=cores

scratch_dir=${HOME}/ml/scratch/job-${SLURM_JOB_ID}
mkdir $scratch_dir
cp system_profile.py $scratch_dir
cd $scratch_dir
echo $PWD

start_time=$(date +%s)
srun python system_profile.py &> output.txt
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "Elapsed time ${elapsed} s"

cd $SLURM_SUBMIT_DIR
echo $PWD

exit