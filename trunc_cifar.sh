#!/bin/bash
#SBATCH --cpus-per-task=20 
#SBATCH --gres=gpu:volta:1
#SBATCH --output=cosine_lr_noised_mom_7_5_200_epochs.%J.out
#SBATCH --error=cosine_lr_noised_mom_7_5_200_epochs.%J.err
#SBATCH --job-name="cosine_lr_noised_mom_7_5_200_epochs" 

# iterate over weight decay values
for weight_decay in .1 .15 .2 .25 .3
do 
	# iterate over learning rates
	for lr in .01
	do
		python3 trunc_cifar.py --lr $lr --momentum .9 --weight_decay $weight_decay --custom_lr_multiplier cosine --trials 3 --out_dir /home/gridsan/stefanou/cosine_lr_noised_mom_7_5_200_epochs --data_path /home/gridsan/stefanou/data --workers 8 --batch_size 128 --logit_ball 7.5 --should_save_ckpt --save_ckpt_iters 25 --log_iters 1 --epochs 200
	done
done


