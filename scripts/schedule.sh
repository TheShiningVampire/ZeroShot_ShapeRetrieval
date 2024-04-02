#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# python src/train.py trainer.max_epochs=5 logger=csv
# python src/train.py trainer.max_epochs=10 logger=csv

log_dir_path = 


CUDA_VISIBLE_DEVICES=0 python src/train.py logger=wandb trainer.max_epochs=100 model.net.model_choice=0 paths.log_dir="/home/dgxadmin/vinit/logs/shape_features_5layers_0_1/" paths.output_dir="/home/dgxadmin/vinit/logs/shape_features_5layers_0_1/" ckpt_path="/home/dgxadmin/vinit/logs/shape_features_5layers_0/checkpoints/last.ckpt"

CUDA_VISIBLE_DEVICES=1 python src/train.py logger=wandb trainer.max_epochs=100 model.net.model_choice=1 paths.log_dir="/home/dgxadmin/vinit/logs/shape_features_5layers_1_1/" paths.output_dir="/home/dgxadmin/vinit/logs/shape_features_5layers_1_1/" ckpt_path="/home/dgxadmin/vinit/logs/shape_features_5layers_1/checkpoints/last.ckpt"

CUDA_VISIBLE_DEVICES=5 python src/train.py logger=wandb trainer.max_epochs=100 model.net.model_choice=2 paths.log_dir="/home/dgxadmin/vinit/logs/shape_features_5layers_2_1/" paths.output_dir="/home/dgxadmin/vinit/logs/shape_features_5layers_2_1/" ckpt_path="/home/dgxadmin/vinit/logs/shape_features_5layers_2/checkpoints/last.ckpt"

CUDA_VISIBLE_DEVICES=4 python src/train.py logger=wandb trainer.max_epochs=100 paths.log_dir="/home/dgxadmin/vinit/logs/final_test_1/" paths.output_dir="/home/dgxadmin/vinit/logs/final_test_1/" 


CUDA_VISIBLE_DEVICES=2 python src/train.py logger=wandb trainer.max_epochs=100 paths.log_dir="/home/dgxadmin/vinit/logs/final_1/" paths.output_dir="/home/dgxadmin/vinit/logs/final_1/" 

CUDA_VISIBLE_DEVICES=2 python src/train.py trainer.max_epochs=100 paths.log_dir="/home/dgxadmin/vinit/logs/temp_1/" paths.output_dir="/home/dgxadmin/vinit/logs/temp_1/" logger=wandb

CUDA_VISIBLE_DEVICES=1 python src/train.py logger=wandb trainer.max_epochs=100 paths.log_dir="/home/dgxadmin/vinit/logs/final_2/" paths.output_dir="/home/dgxadmin/vinit/logs/final_2/" ckpt_path="/home/dgxadmin/vinit/logs/final_1/checkpoints/last.ckpt"

CUDA_VISIBLE_DEVICES=1 python src/train.py logger=wandb trainer.max_epochs=100 paths.log_dir="/home/dgxadmin/vinit/logs/final_3/" paths.output_dir="/home/dgxadmin/vinit/logs/final_3/" 
