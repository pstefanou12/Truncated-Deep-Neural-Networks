#!/bin/bash
#SBATCH --cpus-per-task=20 
#SBATCH --gres=gpu:volta:1
#SBATCH --output=cifar_10_1_trunc.%J.out
#SBATCH --error=cifar_10_1_trunc.%J.err
#SBATCH --job-name="cifar_10_1_trunc" 

# iterate over truncation set
for norm in 3.0 3.5 4.0 5.0
do
	# iterate over weight decay
	for wd in .0005 .05	
	do
		for lr in 1e-2 1e-1
		do
		python3 cifar_10_1.py --lr $lr --momentum .9 --weight_decay $wd --step_lr 50 --step_lr_gamma .1 --trials 3 --out_dir /home/gridsan/stefanou/no_scaling_in_grad --data_path /home/gridsan/stefanou/data --workers 8 --batch_size 128 --logit_ball $norm --should_save_ckpt --save_ckpt_iters 50 --log_iters 5
		done
	done
done


