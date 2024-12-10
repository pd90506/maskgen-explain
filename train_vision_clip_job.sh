#!/bin/bash
#$ -M dpan@nd.edu      # Email address for job notification
#$ -m abe               # Send mail when job begins, ends and aborts
#$ -q gpu@@lucy              # Specify queue
#$ -l gpu=1
#$ -pe smp 8           # Specify number of cores to use.
#$ -N Train_vision_mask_gen        # Specify job name

module load conda

python train_vision_maskgen.py
