#!/bin/bash
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=nuel-sim
#SBATCH --output=output/slurm.%N.%j.out
#SBATCH --error=output/slurm.%N.%j.err

# Create output directory if it doesn't exist
mkdir -p output

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# Run the training script
python -m dqn_marl.train --episodes 5000 --plot --evaluate