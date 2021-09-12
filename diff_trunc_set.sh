#!/bin/bash
#SBATCH --cpus-per-task=20 
#SBATCH --gres=gpu:volta:1
#SBATCH --output=diff_trunc_set_trunc.%J.out
#SBATCH --error=diff_trunc_set_trunc.%J.err
#SBATCH --job-name="diff_trunc_set_trunc" 

# iterate over truncation set
for eps in .001 .01 .05 .1 .5
do
	python3 diff_trunc_set.py --lr 1e-1 --momentum .9 --weight_decay 5e-4 --step_lr 50 --step_lr_gamma .1 --trials 3 --out_dir /home/gridsan/stefanou/diff_trunc_set --data_path /home/gridsan/stefanou/data --workers 8 --batch_size 128 --epsilon $eps --should_save_ckpt --save_ckpt_iters 50 --log_iters 5
done


