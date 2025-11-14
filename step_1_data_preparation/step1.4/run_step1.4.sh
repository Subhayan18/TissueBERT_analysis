#!/bin/bash
#SBATCH --job-name=step1.4_training_dataset
#SBATCH --output=step1.4_%j.out
#SBATCH --error=step1.4_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=240G

# Load modules
module load GCC/12.3.0
module load SciPy-bundle/2023.07
module load matplotlib/3.7.2
module load OpenMPI/4.1.5
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load JupyterNotebook/7.0.2
module load scikit-build/0.17.6
module load imageio/2.33.1
module load Python/3.11.3

# Verify modules loaded
echo "Loaded modules:"
module list

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $PWD"

# Navigate to script directory
cd /home/chattopa/data_storage/TissueBERT_analysis/step_1_data_preparation/step1.4

echo "Script directory: $(pwd)"
echo "Data will be written to: /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset"
echo ""

# Run the script
echo "Starting Step 1.4: Create Training Dataset Structure"
python3 create_training_dataset.py

# Print completion info
echo "End time: $(date)"
echo "Job completed!"
