"""
Configuration file for Step 1.4: Create Training Dataset Structure
All paths and parameters are centralized here for easy modification.
"""

import os

# ============================================================================
# DIRECTORY CONFIGURATION
# ============================================================================

# Scripts will be executed from this directory:
SCRIPT_DIR = "/home/chattopa/data_storage/TissueBERT_analysis/step_1_data_preparation/step1.4"

# Data directories remain in the original location:
DATA_BASE_DIR = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis"

# ============================================================================
# INPUT PATHS
# ============================================================================

# Input files from previous steps (all in MethAtlas_WGBSanalysis)
BETA_MATRIX_FILE = os.path.join(DATA_BASE_DIR, "panel_beta_matrix.tsv")
SYNTHETIC_READS_DIR = os.path.join(DATA_BASE_DIR, "synthetic_reads")
THREEMER_SEQUENCES_FILE = os.path.join(DATA_BASE_DIR, "TWIST_panel_sequences_3mers.txt")
PANEL_BED_FILE = os.path.join(DATA_BASE_DIR, "TWIST_blocks.bed")
TISSUE_LABELS_FILE = os.path.join(DATA_BASE_DIR, "synthetic_reads", "tissue_labels.txt")

# ============================================================================
# OUTPUT PATHS
# ============================================================================

# Main output directory for training dataset (stays in MethAtlas_WGBSanalysis)
OUTPUT_DIR = os.path.join(DATA_BASE_DIR, "training_dataset")

# Subdirectories for train/val/test splits (will be created in Step 1.5)
# For now, we'll create all data in a single directory
TRAIN_DIR = os.path.join(OUTPUT_DIR, "all_data")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.csv")

# Logs and progress tracking
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
PROGRESS_FILE = os.path.join(LOG_DIR, "progress.txt")

# ============================================================================
# DATA AUGMENTATION PARAMETERS
# ============================================================================

# Number of augmented versions per original sample-region pair
N_AUGMENTATIONS = 5

# Augmentation strategy for each version
AUGMENTATION_CONFIG = [
    {
        "version": 0,
        "description": "original",
        "jitter": False,
        "jitter_percent": 0.0,
        "coverage": 500,
        "strand_flip": False
    },
    {
        "version": 1,
        "description": "jitter_100x",
        "jitter": True,
        "jitter_percent": 0.05,
        "coverage": 100,
        "strand_flip": False
    },
    {
        "version": 2,
        "description": "jitter_50x",
        "jitter": True,
        "jitter_percent": 0.05,
        "coverage": 50,
        "strand_flip": False
    },
    {
        "version": 3,
        "description": "jitter_30x",
        "jitter": True,
        "jitter_percent": 0.05,
        "coverage": 30,
        "strand_flip": False
    },
    {
        "version": 4,
        "description": "jitter_10x",
        "jitter": True,
        "jitter_percent": 0.05,
        "coverage": 10,
        "strand_flip": False
    }
]

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Number of tissue types in the dataset
# This will be updated after loading tissue_labels.txt
N_TISSUES = 119  # Updated from tissue_labels.txt

# Sequence parameters
MAX_SEQUENCE_LENGTH = 150  # Maximum read length in bp
KMER_SIZE = 3  # 3-mer tokenization

# 3-mer vocabulary size
# 64 DNA 3-mers (4^3) + 5 special tokens
VOCAB_SIZE = 69
SPECIAL_TOKENS = {
    'PAD': 64,
    'UNK': 65,
    'MASK': 66,
    'CLS': 67,
    'SEP': 68
}

# Methylation states
METH_UNMETHYLATED = 0
METH_METHYLATED = 1
METH_NO_CPG = 2

# ============================================================================
# PROCESSING PARAMETERS
# ============================================================================

# Parallel processing
N_CORES = 40  # Use all available cores
CHUNK_SIZE = 100  # Number of regions to process per worker

# Memory management
MAX_MEMORY_GB = 240  # Maximum memory available

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# VALIDATION PARAMETERS
# ============================================================================

# Quality checks
MIN_COVERAGE_THRESHOLD = 5  # Minimum coverage to include a region
MAX_BETA_VALUE = 1.0  # Maximum valid beta value
MIN_BETA_VALUE = 0.0  # Minimum valid beta value

# ============================================================================
# LOGGING PARAMETERS
# ============================================================================

# Logging level
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR

# Progress reporting frequency
REPORT_EVERY_N_SAMPLES = 10  # Report progress every N samples
REPORT_EVERY_N_REGIONS = 1000  # Report progress every N regions

# ============================================================================
# FILE FORMAT PARAMETERS
# ============================================================================

# NPZ compression
NPZ_COMPRESSION = True  # Use compressed npz format

# CSV parameters
CSV_DELIMITER = ","

# ============================================================================
# TISSUE LABEL MAPPING
# ============================================================================

# This will be populated dynamically from tissue_labels.txt
# Format: {tissue_name: index}
TISSUE_TO_INDEX = {}

# Reverse mapping: {index: tissue_name}
INDEX_TO_TISSUE = {}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_output_directories():
    """Create all necessary output directories."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"Created output directories:")
    print(f"  - {OUTPUT_DIR}")
    print(f"  - {TRAIN_DIR}")
    print(f"  - {LOG_DIR}")

def validate_input_files():
    """Validate that all required input files exist."""
    required_files = {
        "Beta matrix": BETA_MATRIX_FILE,
        "3-mer sequences": THREEMER_SEQUENCES_FILE,
        "Panel BED": PANEL_BED_FILE,
        "Tissue labels": TISSUE_LABELS_FILE
    }
    
    missing_files = []
    for name, path in required_files.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if not os.path.exists(SYNTHETIC_READS_DIR):
        missing_files.append(f"Synthetic reads directory: {SYNTHETIC_READS_DIR}")
    
    if missing_files:
        print("ERROR: Missing required input files:")
        for f in missing_files:
            print(f"  - {f}")
        return False
    
    print("All required input files found!")
    return True

def load_tissue_mapping():
    """Load tissue label mapping from file."""
    global TISSUE_TO_INDEX, INDEX_TO_TISSUE
    
    with open(TISSUE_LABELS_FILE, 'r') as f:
        tissues = [line.strip() for line in f if line.strip()]
    
    # Get unique tissue types
    unique_tissues = sorted(list(set(tissues)))
    
    # Create mapping: each unique tissue name gets an index
    TISSUE_TO_INDEX = {tissue: idx for idx, tissue in enumerate(unique_tissues)}
    INDEX_TO_TISSUE = {idx: tissue for tissue, idx in TISSUE_TO_INDEX.items()}
    
    print(f"Loaded {len(unique_tissues)} unique tissue types from {len(tissues)} samples")
    return TISSUE_TO_INDEX, INDEX_TO_TISSUE

def print_config_summary():
    """Print a summary of the configuration."""
    print("\n" + "="*80)
    print("STEP 1.4: CREATE TRAINING DATASET STRUCTURE")
    print("="*80)
    print("\nCONFIGURATION SUMMARY:")
    print(f"  Script directory: {SCRIPT_DIR}")
    print(f"  Data directory: {DATA_BASE_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"\nAUGMENTATION STRATEGY:")
    print(f"  Number of augmentations: {N_AUGMENTATIONS}x")
    for aug in AUGMENTATION_CONFIG:
        print(f"    v{aug['version']}: {aug['description']} - "
              f"{'Jitter ±' + str(int(aug['jitter_percent']*100)) + '%' if aug['jitter'] else 'No jitter'}, "
              f"{aug['coverage']}x coverage")
    print(f"\nPROCESSING:")
    print(f"  CPU cores: {N_CORES}")
    print(f"  Max memory: {MAX_MEMORY_GB} GB")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"\nEXPECTED OUTPUT:")
    print(f"  Estimated training examples: ~{119 * 51089 * N_AUGMENTATIONS:,}")
    print(f"  Estimated storage: ~{60 * N_AUGMENTATIONS} GB")
    print("="*80 + "\n")

if __name__ == "__main__":
    print_config_summary()
    validate_input_files()
    create_output_directories()
    load_tissue_mapping()
