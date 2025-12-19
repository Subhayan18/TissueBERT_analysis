#!/bin/bash
#SBATCH --job-name=stage2_prep
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/mixture_data/logs/stage2_prep_%j.out
#SBATCH --error=/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/mixture_data/logs/stage2_prep_%j.err

################################################################################
# Stage 2 Dataset Preparation
# 
# Generates blood-masked validation and test mixture datasets
# Must be run BEFORE Stage 2 training
################################################################################

echo "============================================================"
echo "Stage 2: Blood-Masked Dataset Generation"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================================"

# Configuration
WORK_DIR="/home/chattopa/data_storage/TissueBERT_analysis/step_4_mixture_augmentation/bloodmagic"
HDF5_PATH="/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/methylation_dataset.h5"
METADATA_PATH="/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/combined_metadata.csv"
OUTPUT_DIR="/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/mixture_data"

echo ""
echo "Configuration:"
echo "  Working directory: $WORK_DIR"
echo "  HDF5: $HDF5_PATH"
echo "  Metadata: $METADATA_PATH"
echo "  Output: $OUTPUT_DIR"
echo ""

# Verify input files exist
if [ ! -f "$HDF5_PATH" ]; then
    echo "ERROR: HDF5 file not found at $HDF5_PATH"
    exit 1
fi

if [ ! -f "$METADATA_PATH" ]; then
    echo "ERROR: Metadata file not found at $METADATA_PATH"
    exit 1
fi

echo "✓ Input files verified"

# Create output directory
mkdir -p $OUTPUT_DIR/logs

# Change to working directory
cd $WORK_DIR || exit 1

# Load modules
echo "Loading modules..."
source /home/chattopa/data_storage/TissueBERT_analysis/step_2_model_architecture/LMOD.sourceme

# Python info
echo ""
echo "Python environment:"
echo "  Python: $(which python3)"
echo ""

# Run dataset generation
echo "============================================================"
echo "Generating Stage 2 Blood-Masked Mixtures"
echo "============================================================"
echo ""
echo "Strategy:"
echo "  - Generate 1500 validation + 1500 test mixtures"
echo "  - Input: 60-100% blood + other tissues (mixed methylation)"
echo "  - Output: Blood-masked labels (21 tissues, renormalized)"
echo ""

python3 generate_stage2_mixtures.py \
    --hdf5 $HDF5_PATH \
    --metadata $METADATA_PATH \
    --output_dir $OUTPUT_DIR

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Dataset Generation Complete"
echo "============================================================"
echo "Exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "============================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓✓✓ Stage 2 datasets generated successfully! ✓✓✓"
    echo ""
    echo "Generated files:"
    echo "  $OUTPUT_DIR/stage2_validation_mixtures.h5"
    echo "  $OUTPUT_DIR/stage2_test_mixtures.h5"
    echo ""
    echo "Next step:"
    echo "  Submit Stage 2 training: sbatch submit_stage2.sh"
    echo ""
else
    echo ""
    echo "✗ Dataset generation failed with exit code $EXIT_CODE"
    echo "  Check logs for details"
fi

exit $EXIT_CODE
