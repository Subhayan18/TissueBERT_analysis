#!/bin/bash
#
# Step 1.5a - Update Metadata with Clean Tissue Types
# ====================================================
#
# This script updates metadata.csv with:
# - Clean tissue names (e.g., Liver-Hepatocytes_3 -> Liver)
# - Reassigned tissue indices (0, 1, 2, ...)
#
# Usage:
#   bash run_update_metadata.sh [--preview]
#
# Options:
#   --preview    Preview changes without saving files
#

# Set paths
SCRIPT_DIR="/home/chattopa/data_storage/MethAtlas_WGBSanalysis"
METADATA_IN="${SCRIPT_DIR}/metadata.csv"
METADATA_OUT="${SCRIPT_DIR}/updated_metadata.csv"
PYTHON_SCRIPT="update_metadata_tissues.py"

# Check if preview mode
PREVIEW_FLAG=""
if [ "$1" == "--preview" ]; then
    PREVIEW_FLAG="--preview"
    echo "Running in PREVIEW mode - no files will be written"
    echo ""
fi

# Check if metadata.csv exists
if [ ! -f "$METADATA_IN" ]; then
    echo "Error: metadata.csv not found at $METADATA_IN"
    echo "Please check the path and try again."
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found"
    echo "Please ensure the script is in the current directory."
    exit 1
fi

echo "========================================================================"
echo "Step 1.5a: Update Metadata with Clean Tissue Types"
echo "========================================================================"
echo ""
echo "Input:  $METADATA_IN"
echo "Output: $METADATA_OUT"
echo ""

# Run the Python script
python $PYTHON_SCRIPT --input "$METADATA_IN" --output "$METADATA_OUT" $PREVIEW_FLAG

if [ $? -eq 0 ] && [ -z "$PREVIEW_FLAG" ]; then
    echo ""
    echo "========================================================================"
    echo "SUCCESS!"
    echo "========================================================================"
    echo "Updated metadata saved to: $METADATA_OUT"
    echo "Tissue index mapping saved to: ${SCRIPT_DIR}/tissue_index_mapping.csv"
    echo ""
    echo "Next steps:"
    echo "1. Review the updated_metadata.csv file"
    echo "2. Check tissue_index_mapping.csv for tissue->index mapping"
    echo "3. If everything looks good, proceed to Step 1.5b (data splitting)"
else
    if [ -n "$PREVIEW_FLAG" ]; then
        echo ""
        echo "Preview complete. Re-run without --preview to save changes."
    else
        echo ""
        echo "Script failed. Please check the error messages above."
    fi
fi
