#! /bin/bash

#SBATCH --job-name=UCB
#SBATCH --output=job_log/result-%A_%a.out
#SBATCH --error=job_log/result-%A_%a.er
#SBATCH --partition=longq
#SBATCH --time=01-10:00:00

python3 -m main
