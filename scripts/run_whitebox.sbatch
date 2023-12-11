#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1
#SBATCH --job-name test
#SBATCH --partition compute
#SBATCH --time 4:00:00
# current: 897547
CUDA_VISIBLE_DEVICES=0 python minigpt_visual_attack.py \
--cfg_path eval_configs/minigpt4_eval.yaml --gpu_id 0 --n_iters 5000 --alpha 1 --save_dir visual_unconstrained
wait
