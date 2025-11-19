#!/bin/bash
#SBATCH --job-name=cosmos_a40_test
#SBATCH --account=SNIC2024-22-358  # Replace with your actual project account
#SBATCH --partition=gpua40         # A40 GPU partition (adjust based on actual partition name)
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=cosmos_a40_test_%j.log
#SBATCH --error=cosmos_a40_test_%j.err

echo "=================================================="
echo "COSMOS A40 GPU Access Test"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Date: $(date)"
echo ""

nvidia-smi

echo ""
echo "Test completed successfully!"
