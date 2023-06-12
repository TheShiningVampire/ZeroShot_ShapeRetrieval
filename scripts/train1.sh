#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python src/train.py logger=wandb

# Wait for this process to finish
wait

# Run the next process
CUDA_VISIBLE_DEVICES=4 bash /raid/biplab/phduser3/vinit/.system_logs/gpu.sh 1
