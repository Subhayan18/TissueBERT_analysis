#!/bin/bash
#SBATCH --job-name=step1.3_sequences
#SBATCH --output=logs/step1.3_%j.out
#SBATCH --error=logs/step1.3_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Step 1.3: Extract DNA Sequences and Create 3-mer Tokenization
# SLURM submission script for HPC cluster with LMOD

echo "========================================"
echo "Step 1.3: Sequence Extraction Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "========================================"

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules
echo "Loading LMOD modules..."
module reset
module load GCC/12.3.0
module load Python/3.11.3
module load BEDTools/2.31.0

# Verify modules loaded
echo "Loaded modules:"
module list

# Input files
BED_FILE="/home/chattopa/data_storage/MethAtlas_WGBSanalysis/TWIST_blocks.bed"
FASTA_FILE="/home/chattopa/data_storage/wgbs_tools/references/hg38/hg38.fa.gz"

# Output directory
OUTPUT_DIR="./step1.3_output"

# Verify input files exist
echo ""
echo "Verifying input files..."
if [ ! -f "$BED_FILE" ]; then
    echo "ERROR: BED file not found: $BED_FILE"
    exit 1
fi
echo "✓ BED file found: $BED_FILE"

if [ ! -f "$FASTA_FILE" ]; then
    echo "ERROR: FASTA file not found: $FASTA_FILE"
    exit 1
fi
echo "✓ FASTA file found: $FASTA_FILE"

# Run the Python script
echo ""
echo "Running sequence extraction..."
python step1.3_extract_sequences_and_tokenize.py \
    --bed "$BED_FILE" \
    --fasta "$FASTA_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --output-prefix "TWIST_panel_sequences"

# Check if script completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Step 1.3 Completed Successfully!"
    echo "Output location: $OUTPUT_DIR"
    echo "Date: $(date)"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "ERROR: Step 1.3 Failed!"
    echo "Check error log: logs/step1.3_${SLURM_JOB_ID}.err"
    echo "========================================"
    exit 1
fi
