#!/bin/bash
#SBATCH --job-name=quick_gpu_test
#SBATCH --account=SNIC2024-22-358  # Replace with your actual project account
#SBATCH --partition=gpua100        # Try A100 first
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=quick_test_%j.log

echo "Quick GPU Test - Job $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo ""
nvidia-smi
echo ""
echo "Test completed at $(date)"
