#!/bin/bash
#SBATCH --job-name=agrl
#SBATCH --partition=job
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --output=output_%j.log   # 标准输出（print 的内容）
#SBATCH --error=error_%j.log    # 错误输出

bash sciworld_train.sh

bash textcraft_train.sh