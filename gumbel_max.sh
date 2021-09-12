#!/bin/bash
#SBATCH --cpus-per-task=20 
#SBATCH --gres=gpu:volta:1
#SBATCH --output=cifar_10_1_trunc.%J.out
#SBATCH --error=cifar_10_1_trunc.%J.err
#SBATCH --job-name="cifar_10_1_trunc" 

# iterate over truncation set
for norm in 0.0 1.0 2.0 3.0 3.5 4.0 5.0
do
		python3 cifar_10_1.py --lr 1e-1 --momentum .9 --weight_decay .0005 --step_lr 50 --step_lr_gamma .1 --trials 3 --out_dir /home/gridsan/stefanou/gumbel_max --data_path /home/gridsan/stefanou/data --workers 8 --batch_size 128 --logit_ball $norm --should_save_ckpt --save_ckpt_iters 50 --log_iters 5
done


