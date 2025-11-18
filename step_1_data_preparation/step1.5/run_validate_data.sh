#!/bin/bash
#
# Step 1.5d - Final Data Validation
# ==================================
#
# Comprehensive validation of all Step 1.5 outputs before Phase 2.
#
# Usage:
#   bash run_validate_data.sh
#

# Set paths
BASE_DIR="/home/chattopa/data_storage/MethAtlas_WGBSanalysis"
DATA_DIR="${BASE_DIR}/training_dataset/all_data"
SYNTHETIC_DIR="${BASE_DIR}/synthetic_samples"
SPLITS_DIR="${BASE_DIR}/data_splits"
PYTHON_SCRIPT="validate_final_data.py"

# Files to validate
TRAIN_FILES="${SPLITS_DIR}/train_files.csv"
VAL_FILES="${SPLITS_DIR}/val_files.csv"
TEST_FILES="${SPLITS_DIR}/test_files.csv"

# Parameters
SAMPLE_SIZE=50  # Number of files to check for integrity

echo "========================================================================"
echo "Step 1.5d: Final Data Validation"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Data directory:     $DATA_DIR"
echo "  Synthetic directory: $SYNTHETIC_DIR"
echo "  Splits directory:   $SPLITS_DIR"
echo "  Sample size:        $SAMPLE_SIZE files"
echo ""

# Check if split files exist
if [ ! -f "$TRAIN_FILES" ]; then
    echo "Error: train_files.csv not found at $TRAIN_FILES"
    echo "Please run Step 1.5c (data splitting) first."
    exit 1
fi

if [ ! -f "$VAL_FILES" ]; then
    echo "Error: val_files.csv not found at $VAL_FILES"
    echo "Please run Step 1.5c (data splitting) first."
    exit 1
fi

if [ ! -f "$TEST_FILES" ]; then
    echo "Error: test_files.csv not found at $TEST_FILES"
    echo "Please run Step 1.5c (data splitting) first."
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found"
    echo "Please ensure the script is in the current directory."
    exit 1
fi

echo "Starting comprehensive validation..."
echo ""

# Run the Python script
python "$PYTHON_SCRIPT" \
    --data-dir "$DATA_DIR" \
    --train-files "$TRAIN_FILES" \
    --val-files "$VAL_FILES" \
    --test-files "$TEST_FILES" \
    --synthetic-dir "$SYNTHETIC_DIR" \
    --sample-size $SAMPLE_SIZE

exit_code=$?

echo ""
echo "========================================================================"

if [ $exit_code -eq 0 ]; then
    echo "SUCCESS - All validation tests passed!"
    echo "========================================================================"
    echo ""
    echo "✓ Your data is ready for Phase 2: Model Training"
    echo ""
    echo "Summary of validated data:"
    echo "  - Training files:   $TRAIN_FILES"
    echo "  - Validation files: $VAL_FILES"
    echo "  - Test files:       $TEST_FILES"
    echo ""
    echo "Next steps:"
    echo "  1. Proceed to Phase 2 of the roadmap"
    echo "  2. Begin model architecture design"
    echo "  3. Start training your DNABERT-S model"
else
    echo "FAILED - Validation errors found"
    echo "========================================================================"
    echo ""
    echo "✗ Please fix the errors above before proceeding"
    echo ""
    echo "Common issues:"
    echo "  - Missing .npz files"
    echo "  - Corrupted data files"
    echo "  - Inconsistent metadata"
    echo "  - Split integrity problems"
    echo ""
    echo "Review the error messages and re-run previous steps if needed."
fi

exit $exit_code
