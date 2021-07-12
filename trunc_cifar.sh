#!/bin/bash
#SBATCH --cpus-per-task=20 
#SBATCH --gres=gpu:volta:1
#SBATCH --output=adam_noised_weight_decay.%J.out
#SBATCH --error=adam_noised_weight_decay.%J.err
#SBATCH --job-name="adam noised lr - weight decay" 

# parameters to tune
LEARNING_RATES=(.001 .01 .05 .1 .2 .3)
WEIGHT_DECAY=(.00001 .0001 .0005 .001 .005 .02 .05 .1 .2)

# iterate over hyperparameters
for weight_decay in $WEIGHT_DECAY
do 
	for lr in $LEARNING_RATES
	do
		python3 trunc_cifar.py --lr $lr --momentum 0.0 --weight_decay $weight_decay --adam  --trials 10 --out_dir /home/gridsan/stefanou/adam_noised_weight_decay --data_path /home/gridsan/stefanou/data --workers 8 --batch_size 128 --logit_ball 5.0 --should_save_ckpt --save_ckpt_iters 25 --log_iters 1
	done
done


