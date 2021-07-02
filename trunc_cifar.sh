#!/bin/bash

# Slurm sbatch options

# Loading the required module
source /etc/profile
module load anaconda/2021a

# Run the script
python3 trunc_cifar.py

