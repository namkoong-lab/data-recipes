#!/bin/bash

CUDA_DEVICES='0,1,2,3'

# Check if all arguments are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 scale start_num end_num [master_port]"
    echo "Default master_port is 29501 if not specified"
    exit 1
fi

scale=$1
start_num=$2
end_num=$3
nproc=$(echo $CUDA_DEVICES | tr ',' '\n' | wc -l)
master_port=${4:-29501}

cd /mnt/home/tyen/olmo-data-recipe

# Loop through the range
for ((i=start_num; i<=end_num; i++)); do
    # Format number with leading zeros (0000, 0001, etc.)
    config_num=$(printf "%04d" $i)

    # Construct the command
    cmd="CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m torch.distributed.run --nproc_per_node=$nproc --master_port=$master_port scripts/train.py ~/data-recipes/data_mixing_experiments/olmo/configs/$scale/config_$config_num.yaml"

    # Echo and execute the command
    echo "Executing: $cmd"
    eval "$cmd"
    wait $!
done
