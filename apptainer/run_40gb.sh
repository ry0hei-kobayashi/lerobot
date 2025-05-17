#!/bin/bash
#SBATCH --gres=gpu:a100_3g.40gb:1

#apptainer exec --nvccli pg-blip.sif nvidia-smi
sh singularity_run.sh
