#!/bin/bash
#
# Step 1.5c - Train/Validation/Test Splitting
# ============================================
#
# This script splits samples into train/validation/test sets with:
# - Stratification by tissue type
# - 70% train / 15% validation / 15% test
# - All augmentations stay with their sample
#
# Usage:
#   bash run_split_data.sh
#

# Set paths
BASE_DIR="/home/chattopa/data_storage/MethAtlas_WGBSanalysis"
METADATA="${BASE_DIR}/synthetic_samples/combined_metadata.csv"
OUTPUT_DIR="${BASE_DIR}/data_splits"
PYTHON_SCRIPT="split_train_val_test.py"

# Parameters
TRAIN_RATIO=0.70
VAL_RATIO=0.15
TEST_RATIO=0.15
RANDOM_SEED=42

echo "========================================================================"
echo "Step 1.5c: Train/Validation/Test Splitting"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Metadata:      $METADATA"
echo "  Output dir:    $OUTPUT_DIR"
echo "  Train ratio:   $TRAIN_RATIO (70%)"
echo "  Val ratio:     $VAL_RATIO (15%)"
echo "  Test ratio:    $TEST_RATIO (15%)"
echo "  Random seed:   $RANDOM_SEED"
echo ""

# Check if metadata exists
if [ ! -f "$METADATA" ]; then
    echo "Error: Metadata file not found at $METADATA"
    echo "Please run Step 1.5b (synthetic generation) first."
    echo ""
    echo "If you didn't generate synthetic samples, use:"
    echo "  METADATA=\"${BASE_DIR}/updated_metadata.csv\""
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found"
    echo "Please ensure the script is in the current directory."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting data splitting..."
echo ""

# Run the Python script
python "$PYTHON_SCRIPT" \
    --metadata "$METADATA" \
    --output-dir "$OUTPUT_DIR" \
    --train-ratio $TRAIN_RATIO \
    --val-ratio $VAL_RATIO \
    --test-ratio $TEST_RATIO \
    --seed $RANDOM_SEED

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "SUCCESS!"
    echo "========================================================================"
    echo ""
    echo "Split files saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    echo "  - train_files.csv       (training file list)"
    echo "  - val_files.csv         (validation file list)"
    echo "  - test_files.csv        (test file list)"
    echo "  - train_samples.txt     (training sample names)"
    echo "  - val_samples.txt       (validation sample names)"
    echo "  - test_samples.txt      (test sample names)"
    echo "  - split_config.json     (split configuration)"
    echo ""
    echo "Next steps:"
    echo "  1. Review the split summary above"
    echo "  2. Check train_files.csv, val_files.csv, test_files.csv"
    echo "  3. Proceed to Phase 2: Model Training"
else
    echo ""
    echo "========================================================================"
    echo "ERROR: Script failed"
    echo "========================================================================"
    echo "Please check the error messages above."
    exit 1
fi
