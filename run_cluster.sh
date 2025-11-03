#!/bin/bash
#SBATCH --partition=stampede
#SBATCH --job-name=nuel-sim
#SBATCH --output=/home-mscluster/dngobe/nuel-sim-1.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# Run multiple training sessions for convergence analysis
python -m scripts.run_multiple_training --runs 5 --episodes 4000 --output-dir multiple_runs

# Analyze convergence
python -m scripts.analyze_convergence --runs-dir multiple_runs
