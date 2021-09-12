#!/bin/bash
#SBATCH --cpus-per-task=20 
#SBATCH --gres=gpu:volta:1
#SBATCH --output=identity.%J.out
#SBATCH --error=identity.%J.err
#SBATCH --job-name="identity" 

# iterate over weight decay
for wd in .0005	
do
	for lr in 1e-1
	do
	python3 cifar_10_1.py --lr $lr --momentum .9 --weight_decay $wd --step_lr 50 --step_lr_gamma .1 --trials 3 --out_dir /home/gridsan/stefanou/identity --data_path /home/gridsan/stefanou/data --workers 8 --batch_size 128 --should_save_ckpt --save_ckpt_iters 50 --log_iters 5
	done
done


