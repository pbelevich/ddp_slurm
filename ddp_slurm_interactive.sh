#!/bin/bash

srun --label \
	--job-name=ddp_slurm_interactive \
	--ntasks=16 \
	--partition=dev \
	--nodes=2 \
	--gpus-per-node=8 \
	--gpus-per-task=1 \
	--time=1:00:00 \
	ddp_slurm.sh

