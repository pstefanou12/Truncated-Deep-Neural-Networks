#!/bin/bash
#SBATCH --cpus-per-task=20 
#SBATCH --gres=gpu:volta:1
# Run the script
python3 trunc_cifar.py --lr 1e-1 --momentum .9 --weight_decay 5e-4 --custom_lr_multiplier cyclic --trials 10 --out_dir /home/gridsan/stefanou/cyclic --data_path /home/gridsan/stefanou/data --workers 10 --batch_size 128 --logit_ball 5.0 --should_save_ckpt --save_ckpt_iters 25 --log_iters 1
