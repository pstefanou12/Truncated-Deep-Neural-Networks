#!/bin/bash
#SBATCH --cpus-per-task=20 
#SBATCH --gres=gpu:volta:1
#SBATCH --output=cyclic_log_tune.%J.out
#SBATCH --error=cyclic_log_tune.%J.err
#SBATCH --job-name="cyclic_log_tune" 

# iterate over weight decay values
for weight_decay in .0005 .001 .005 .01 .05
do 
	# iterate over logit balls
	for norm in 5 5.5 6.0 6.5 7.0 7.5 10.0 
	do
			python3 trunc_cifar.py --lr .01 --momentum .9 --weight_decay $weight_decay --custom_lr_multiplier cyclic --trials 3 --out_dir /home/gridsan/stefanou/cyclic_log_tune --data_path /home/gridsan/stefanou/data --workers 8 --batch_size 128 --logit_ball $norm --should_save_ckpt --save_ckpt_iters 1 --log_iters 1
	done
done


