#!/bin/bash
#SBATCH --job-name=step1.4_TEST
#SBATCH --output=step1.4_test_%j.out
#SBATCH --error=step1.4_test_%j.err
#SBATCH --time=06:00:00          # 2 hours should be plenty for test
#SBATCH --cpus-per-task=10
#SBATCH --mem=250G

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

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Navigate to script directory
cd /home/chattopa/data_storage/TissueBERT_analysis/step_1_data_preparation/step1.4

echo "Running TEST mode (5 samples)"

# Run test
python3 create_training_dataset.py --test

echo "End time: $(date)"
echo "Test completed!"
