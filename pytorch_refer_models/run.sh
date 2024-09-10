#!/bin/bash
#SBATCH -o Conv_tasnetgcn_frame_tra_A10_20221022.out
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa12
#SBATCH -p new
python train.py
#python evaluate.py
#python separate.py


nohup python preprocess.py >Conv_tasnetgcn_fea-learnableA_tra_20230208.out 2>&1 &

nohup python train.py >Conv_tasnetgcn_fea-learnableA_tra_20230414.out 2>&1 &
nohup python evaluate.py >Conv_tasnetgcn_fea-learnableA_eva_20230208.out 2>&1 &

nohup python train.py >mtfaa_tra_20240514.out 2>&1 &
nohup python eval_on_dns.py >mtfaa_eva_20240514.out 2>&1 &


nohup python train.py >gunet-tra_20240712.out 2>&1 &
nohup python eval_on_dns.py >gunet-eva_20240712.out 2>&1 &