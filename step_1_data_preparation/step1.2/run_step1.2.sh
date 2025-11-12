#!/bin/bash
#SBATCH --job-name=step1.2_simulate_reads
#SBATCH --output=logs/step1.2_%j.out
#SBATCH --error=logs/step1.2_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=240G

################################################################################
# SLURM Job Script: Step 1.2 - Simulate Read-Level Training Data
################################################################################
#
# Purpose: 
#   Generate synthetic 150bp cfDNA reads from panel_beta_matrix.tsv
#   This creates the training data needed for DNABERT-S model
#
# What this script does:
#   1. Sets up the computing environment (loads modules, activates conda)
#   2. Runs three Python scripts in sequence:
#      - 01_inspect_data.py:    Load and validate input data
#      - 02_simulate_reads.py:  Generate synthetic reads (main work)
#      - 03_verify_output.py:   Quality check the results
#
# Requirements:
#   - LMOD modules: GCC/12.3.0, SciPy-bundle/2023.07, Python/3.11.3
#   - Input file: panel_beta_matrix.tsv from Step 1.1
#   - ~250GB RAM, 48 cores, 72 hours walltime
#
# Usage:
#   sbatch run_step1.2.sh
#
# Output:
#   - /home/chattopa/data_storage/MethAtlas_WGBSanalysis/synthetic_reads/
#     ├── sample_000_Neuron.npz
#     ├── sample_001_Cortex-Neuron_1.npz
#     ├── ...
#     ├── tissue_labels.txt
#     ├── simulation_summary.txt
#     └── sample_statistics.csv
#
################################################################################

# Exit on any error
set -e

# Print start time and node information
echo "========================================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================================================"
echo ""

################################################################################
# SECTION 1: Environment Setup
################################################################################

echo "SECTION 1: Setting up environment"
echo "------------------------------------------------------------------------"

# Create logs directory if it doesn't exist
mkdir -p logs

# Reset modules to start clean
echo "Resetting modules..."
module reset
echo "✓ Modules reset"
echo ""

# Load required LMOD modules
echo "Loading required LMOD modules..."
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
if ! command -v python &> /dev/null; then
    echo "ERROR: python command not found after loading modules"
    exit 1
fi
echo "✓ All modules loaded successfully"
echo ""

# Display loaded modules
echo "Currently loaded modules:"
module list
echo ""

# Print Python and package versions for reproducibility
echo "Software versions:"
python --version
echo "Numpy version: $(python -c 'import numpy; print(numpy.__version__)')"
echo "Pandas version: $(python -c 'import pandas; print(pandas.__version__)')"
echo "Scipy version: $(python -c 'import scipy; print(scipy.__version__)')"
echo ""

################################################################################
# SECTION 2: Run Script 01 - Inspect Data
################################################################################

echo "========================================================================"
echo "SECTION 2: Running Script 01 - Inspect Data"
echo "========================================================================"
echo ""

# Define script location (modify if your scripts are in a different directory)
SCRIPT_DIR="/home/chattopa/data_storage/TissueBERT_analysis/step_1_data_preparation/step1.2"

# Check if script exists
if [[ ! -f "$SCRIPT_DIR/01_inspect_data.py" ]]; then
    echo "ERROR: Script not found: $SCRIPT_DIR/01_inspect_data.py"
    echo "Please ensure scripts are in the correct location"
    exit 1
fi

# Run the script
echo "Executing: python $SCRIPT_DIR/01_inspect_data.py"
echo ""

python "$SCRIPT_DIR/01_inspect_data.py"

# Check if script succeeded
if [[ $? -ne 0 ]]; then
    echo "ERROR: Script 01 failed"
    exit 1
fi

echo ""
echo "✓ Script 01 completed successfully"
echo ""

################################################################################
# SECTION 3: Run Script 02 - Simulate Reads (Main Work)
################################################################################

echo "========================================================================"
echo "SECTION 3: Running Script 02 - Simulate Reads"
echo "========================================================================"
echo ""
echo "⚠ NOTE: This is the main computation step and will take several hours"
echo ""

# Check if script exists
if [[ ! -f "$SCRIPT_DIR/02_simulate_reads.py" ]]; then
    echo "ERROR: Script not found: $SCRIPT_DIR/02_simulate_reads.py"
    exit 1
fi

# Record start time for this section
START_TIME=$(date +%s)

# Run the script
echo "Executing: python $SCRIPT_DIR/02_simulate_reads.py"
echo ""

python "$SCRIPT_DIR/02_simulate_reads.py"

# Check if script succeeded
if [[ $? -ne 0 ]]; then
    echo "ERROR: Script 02 failed"
    exit 1
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_HOURS=$((ELAPSED / 3600))
ELAPSED_MINS=$(((ELAPSED % 3600) / 60))

echo ""
echo "✓ Script 02 completed successfully"
echo "Time elapsed: ${ELAPSED_HOURS}h ${ELAPSED_MINS}m"
echo ""

################################################################################
# SECTION 4: Run Script 03 - Verify Output
################################################################################

echo "========================================================================"
echo "SECTION 4: Running Script 03 - Verify Output"
echo "========================================================================"
echo ""

# Check if script exists
if [[ ! -f "$SCRIPT_DIR/03_verify_output.py" ]]; then
    echo "ERROR: Script not found: $SCRIPT_DIR/03_verify_output.py"
    exit 1
fi

# Run the script
echo "Executing: python $SCRIPT_DIR/03_verify_output.py"
echo ""

python "$SCRIPT_DIR/03_verify_output.py"

# Check if script succeeded
if [[ $? -ne 0 ]]; then
    echo "ERROR: Script 03 failed"
    exit 1
fi

echo ""
echo "✓ Script 03 completed successfully"
echo ""

################################################################################
# SECTION 5: Final Summary
################################################################################

echo "========================================================================"
echo "ALL SCRIPTS COMPLETED SUCCESSFULLY"
echo "========================================================================"
echo ""
echo "Output directory: /home/chattopa/data_storage/MethAtlas_WGBSanalysis/synthetic_reads/"
echo ""
echo "Generated files:"
echo "  - 82 NPZ files (sample_*.npz) containing synthetic reads"
echo "  - tissue_labels.txt: simplified tissue names"
echo "  - simulation_summary.txt: detailed report"
echo "  - sample_statistics.csv: per-sample metrics"
echo ""
echo "Next step: Step 1.3 - Add DNA Sequence Context"
echo ""
echo "Job finished at: $(date)"
echo "========================================================================"

# Print resource usage
echo ""
echo "Resource usage summary:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,MaxRSS,MaxVMSize,State
