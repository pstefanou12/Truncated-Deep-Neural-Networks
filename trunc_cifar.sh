#!/bin/bash
#SBATCH --cpus-per-task=20 
#SBATCH --gres=gpu:volta:1
#SBATCH --output=constant_noised_weight_decay.%J.out
#SBATCH --error=constant_noised_weight_decay.%J.err
#SBATCH --job-name="constant_noised lr - weight decay" 

# iterate over weight decay values
for weight_decay in .00001 .0001 .0005 .001 .005 .02 .05 .1 .2
do 
	# iterate over learning rates
	for lr in .001 .01 .05 .1 .2 .3
	do
	 		python3 trunc_cifar.py --lr $lr --momentum 0.0 --weight_decay $weight_decay --step_lr 10 --step_lr_gamma 1.0  --trials 1 --out_dir /home/gridsan/stefanou/constant_noised_weight_decay --data_path /home/gridsan/stefanou/data --workers 8 --batch_size 128 --logit_ball 5.0 --should_save_ckpt --save_ckpt_iters 25 --log_iters 1
	done
done


