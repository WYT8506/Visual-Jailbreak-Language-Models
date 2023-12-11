#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1
#SBATCH --job-name test
#SBATCH --partition compute
#SBATCH --time 24:00:00
# current: 897547
CUDA_VISIBLE_DEVICES=0 python -u minigpt_blackbox.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0 --image_path  adversarial_images/prompt_unconstrained.bmp
wait