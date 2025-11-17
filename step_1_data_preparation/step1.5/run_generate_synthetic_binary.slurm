#!/bin/bash
#SBATCH --job-name=syn_gen_binary
#SBATCH --output=syn_gen_%A_%a.out
#SBATCH --error=syn_gen_%A_%a.err
#SBATCH --array=0-12           # 13 rare tissues (adjust based on your data)
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G

# SLURM Array Job for Synthetic Sample Generation
# ================================================
#
# This script runs multiple parallel jobs, one per rare tissue.
# Each job processes all augmentations for that tissue's synthetic samples.
#
# Adjust --array parameter based on number of rare tissues:
#   - Check updated_metadata.csv to count tissues with <4 samples
#   - Set --array=0-N where N = (number of rare tissues - 1)
#
# Example: If you have 13 rare tissues, use --array=0-12

# Load modules
module load GCC/12.3.0
module load SciPy-bundle/2023.07
module load matplotlib/3.7.2
module load Python/3.11.3

# Change to working directory
# cd /home/chattopa/data_storage/MethAtlas_WGBSanalysis

echo "========================================================================"
echo "SLURM Array Job: Synthetic Sample Generation"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo ""

# Run the wrapper script (which will use SLURM_ARRAY_TASK_ID)
bash run_generate_synthetic_binary.sh

exit_code=$?

echo ""
echo "Finished: $(date)"
echo "Exit code: $exit_code"

exit $exit_code
