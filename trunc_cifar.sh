#!/bin/bash

# Slurm sbatch options

# Loading the required module
source /etc/profile
module load anaconda/2021a

echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

# Run the script
python3 trunc_cifar.py

