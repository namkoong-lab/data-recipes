#!/bin/bash
#SBATCH -p columbia
#SBATCH -N 1
#SBATCH -c 48
#SBATCH --mem=250000
#SBATCH --gpus=4
#SBATCH --time=24:00:00

# Check if all required arguments are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <scale_type> <start_num> <end_num> [master_port]"
    exit 1
fi

SCALE_TYPE=$1
START_NUM=$2
END_NUM=$3
MASTER_PORT=${4:-29500}  # Default to 29500 if not provided

cd /mnt/home/tyen/data-recipes/data_mixing_experiments/olmo/jobs
scontrol update jobid=$SLURM_JOB_ID name=${SCALE_TYPE}_${START_NUM}_${END_NUM}

bash run_olmo.sh $SCALE_TYPE $START_NUM $END_NUM $MASTER_PORT