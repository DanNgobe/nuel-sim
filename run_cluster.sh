#!/bin/bash
#SBATCH --partition=bigbatch
#SBATCH --time=02:00:00
#SBATCH --job-name=nuel-sim
#SBATCH --output=/homecluster/%u/nuel-sim_%j.out

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the training script
python -m dqn_marl.train --episodes 5000 --plot --evaluate