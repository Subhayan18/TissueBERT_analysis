"""
Main Script for Step 1.4: Create Training Dataset Structure

This script orchestrates the entire process of creating PyTorch-compatible
training data from synthetic reads, beta values, and 3-mer sequences.

Usage:
    python create_training_dataset.py [--test] [--n-samples N]
    
Options:
    --test: Run on a small subset for testing
    --n-samples N: Process only first N samples (for testing)
"""

import numpy as np
import pandas as pd
import os
import sys
import argparse
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging

# Import local modules
import config
from data_loader import DataLoader
from data_augmentation import DataAugmenter


# ============================================================================
# Module-level helper functions for multiprocessing
# ============================================================================

def threemers_to_tokens(threemer_sequence: str, threemer_to_id: dict) -> np.ndarray:
    """Convert space-separated 3-mer sequence to token IDs."""
    threemers = threemer_sequence.split()
    
    token_ids = []
    for kmer in threemers:
        if kmer in threemer_to_id:
            token_ids.append(threemer_to_id[kmer])
        else:
            token_ids.append(config.SPECIAL_TOKENS['UNK'])
    
    return np.array(token_ids, dtype=np.int32)


def create_threemer_vocab():
    """Create 3-mer to ID vocabulary."""
    bases = ['A', 'C', 'G', 'T']
    threemers = []
    
    for b1 in bases:
        for b2 in bases:
            for b3 in bases:
                threemers.append(b1 + b2 + b3)
    
    threemer_to_id = {kmer: i for i, kmer in enumerate(threemers)}
    return threemer_to_id


def aggregate_methylation(methylation_reads: list) -> np.ndarray:
    """Aggregate methylation across multiple reads."""
    if not methylation_reads:
        return np.array([], dtype=np.uint8)
    
    max_len = max(len(read) for read in methylation_reads)
    padded_reads = []
    
    for read in methylation_reads:
        if len(read) < max_len:
            padded = np.full(max_len, config.METH_NO_CPG, dtype=np.uint8)
            padded[:len(read)] = read
            padded_reads.append(padded)
        else:
            padded_reads.append(read)
    
    reads_array = np.array(padded_reads)
    
    aggregated = np.zeros(max_len, dtype=np.uint8)
    for i in range(max_len):
        values = reads_array[:, i]
        unique, counts = np.unique(values, return_counts=True)
        aggregated[i] = unique[np.argmax(counts)]
    
    return aggregated


def pad_or_truncate(array: np.ndarray, target_length: int, pad_value: int) -> np.ndarray:
    """Pad or truncate array to target length."""
    if len(array) == target_length:
        return array
    elif len(array) > target_length:
        return array[:target_length]
    else:
        padded = np.full(target_length, pad_value, dtype=array.dtype)
        padded[:len(array)] = array
        return padded


def convert_to_training_format(methylation_reads: list,
                               threemer_sequence: str,
                               tissue_one_hot: np.ndarray,
                               sample_info: dict,
                               threemer_to_id: dict) -> dict:
    """Convert augmented data to PyTorch-compatible format."""
    dna_tokens = threemers_to_tokens(threemer_sequence, threemer_to_id)
    methylation_pattern = aggregate_methylation(methylation_reads)
    
    dna_tokens = pad_or_truncate(dna_tokens, config.MAX_SEQUENCE_LENGTH, 
                                  pad_value=config.SPECIAL_TOKENS['PAD'])
    methylation_pattern = pad_or_truncate(methylation_pattern, config.MAX_SEQUENCE_LENGTH,
                                          pad_value=config.METH_NO_CPG)
    
    return {
        'dna_tokens': dna_tokens.astype(np.int32),
        'methylation': methylation_pattern.astype(np.uint8),
        'tissue_label': tissue_one_hot.astype(np.float32),
        'n_reads': len(methylation_reads),
        'avg_coverage': len(methylation_reads)
    }


def generate_filename(sample_name: str, region_id: str, aug_version: int) -> str:
    """Generate filename for training file."""
    sample_clean = sample_name.replace('/', '_').replace(' ', '_')
    region_clean = region_id.replace(':', '_').replace('-', '_')
    filename = f"{sample_clean}_{region_clean}_aug{aug_version}.npz"
    return filename


def save_training_file(training_data: dict, filename: str):
    """Save training data to NPZ file."""
    filepath = os.path.join(config.TRAIN_DIR, filename)
    
    if config.NPZ_COMPRESSION:
        np.savez_compressed(filepath, **training_data)
    else:
        np.savez(filepath, **training_data)



def _process_sample_worker(args):
    """
    Worker function for multiprocessing - processes a single sample.
    Creates one file per augmentation version containing all regions.
    """
    sample_name, sample_file_path, threemer_sequences_dict, region_ids_list, tissue_mapping, sample_idx, total_samples = args
    
    # Import here to avoid pickling issues
    import numpy as np
    import config
    from data_augmentation import DataAugmenter
    
    stats = {
        'sample_name': sample_name,
        'n_regions': 0,
        'n_augmentations': 0,
        'n_files': 0,
        'n_failed': 0
    }
    
    try:
        # Load this sample's data
        data = np.load(sample_file_path, allow_pickle=True)
        sample_data = {
            'methylation_data': data['methylation_data'],
            'read_boundaries': data['read_boundaries'],
            'read_lengths': data['read_lengths'],
            'region_indices': data['region_indices'],
            'region_ids': data['region_ids'],
            'read_ids': data['read_ids'],
            'tissue_label': str(data['tissue_label']),
            'reads_per_region': int(data['reads_per_region']),
            'biological_noise': float(data['biological_noise']),
            'read_length': int(data['read_length'])
        }
        
        tissue = sample_data['tissue_label']
        tissue_idx = tissue_mapping[tissue]
        tissue_one_hot = np.zeros(config.N_TISSUES, dtype=np.float32)
        tissue_one_hot[tissue_idx] = 1.0
        
        # Create augmenter
        augmenter = DataAugmenter()
        
        # Create 3-mer vocabulary
        threemer_to_id = create_threemer_vocab()
        
        n_regions = len(sample_data['region_ids'])
        
        # Initialize storage for each augmentation version
        # Dict: {aug_version: {'dna_tokens': [], 'methylation': [], 'region_ids': []}}
        aug_data = {}
        for aug_config in config.AUGMENTATION_CONFIG:
            aug_version = aug_config['version']
            aug_data[aug_version] = {
                'dna_tokens': [],
                'methylation': [],
                'region_ids': [],
                'n_reads': [],
                'aug_config': aug_config
            }
        
        # Process each region
        for region_idx in range(n_regions):
            try:
                # Get region ID from threemer sequences (by index)
                region_id = region_ids_list[region_idx]
                
                # Find reads for this region
                region_mask = sample_data['region_indices'] == region_idx
                read_indices = np.where(region_mask)[0]
                
                methylation_reads = []
                for read_idx in read_indices:
                    start = sample_data['read_boundaries'][read_idx]
                    end = sample_data['read_boundaries'][read_idx + 1]
                    meth_array = sample_data['methylation_data'][start:end]
                    methylation_reads.append(meth_array)
                
                # Get 3-mer sequence
                threemer_seq = threemer_sequences_dict[region_id]['3mer_sequence']
                
                # Create augmented versions
                augmented_versions = augmenter.create_augmented_versions(
                    methylation_reads, threemer_seq
                )
                
                # Add each augmented version to its respective storage
                for aug_reads, aug_threemer, aug_config in augmented_versions:
                    aug_version = aug_config['version']
                    
                    # Convert to training format (but don't save yet)
                    dna_tokens = threemers_to_tokens(aug_threemer, threemer_to_id)
                    methylation_pattern = aggregate_methylation(aug_reads)
                    
                    dna_tokens = pad_or_truncate(dna_tokens, config.MAX_SEQUENCE_LENGTH, 
                                                  pad_value=config.SPECIAL_TOKENS['PAD'])
                    methylation_pattern = pad_or_truncate(methylation_pattern, config.MAX_SEQUENCE_LENGTH,
                                                          pad_value=config.METH_NO_CPG)
                    
                    # Accumulate in arrays
                    aug_data[aug_version]['dna_tokens'].append(dna_tokens)
                    aug_data[aug_version]['methylation'].append(methylation_pattern)
                    aug_data[aug_version]['region_ids'].append(region_id)
                    aug_data[aug_version]['n_reads'].append(len(aug_reads))
                
                stats['n_regions'] += 1
                
            except Exception as e:
                print(f"ERROR region {region_idx} in {sample_name}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                stats['n_failed'] += 1
        
        # Save one file per augmentation version
        for aug_version, data_dict in aug_data.items():
            if len(data_dict['dna_tokens']) > 0:
                # Convert lists to arrays
                combined_data = {
                    'dna_tokens': np.array(data_dict['dna_tokens'], dtype=np.int32),
                    'methylation': np.array(data_dict['methylation'], dtype=np.uint8),
                    'region_ids': np.array(data_dict['region_ids'], dtype='U50'),
                    'n_reads': np.array(data_dict['n_reads'], dtype=np.int32),
                    'tissue_label': tissue_one_hot,
                    'sample_name': sample_name,
                    'tissue_name': tissue
                }
                
                # Generate filename: sample_name_augV.npz
                filename = f"{sample_name}_aug{aug_version}.npz"
                save_training_file(combined_data, filename)
                
                stats['n_files'] += 1
                stats['n_augmentations'] += len(data_dict['dna_tokens'])
        
        return stats
        
    except Exception as e:
        print(f"ERROR loading sample {sample_name}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        stats['n_failed'] = stats['n_regions']
        return stats


class TrainingDatasetCreator:
    """
    Creates PyTorch-compatible training dataset with augmentation.
    """
    
    def __init__(self, test_mode: bool = False, n_samples: int = None):
        """
        Initialize the dataset creator.
        
        Args:
            test_mode: If True, process only a small subset
            n_samples: Number of samples to process (None = all)
        """
        self.test_mode = test_mode
        self.n_samples = n_samples
        
        # Initialize components
        self.loader = DataLoader()
        self.augmenter = DataAugmenter()
        
        # Statistics tracking
        self.stats = {
            'total_samples': 0,
            'total_regions': 0,
            'total_augmentations': 0,
            'total_files_created': 0,
            'failed_regions': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(config.LOG_DIR, 
                                f'training_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        logging.basicConfig(
            level=getattr(logging, config.LOG_LEVEL),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        """Main execution pipeline."""
        self.logger.info("="*80)
        self.logger.info("STEP 1.4: CREATE TRAINING DATASET STRUCTURE")
        self.logger.info("="*80)
        
        self.stats['start_time'] = datetime.now()
        
        try:
            # Step 1: Validate and create directories
            self.logger.info("\n1. Validating configuration and creating directories...")
            config.validate_input_files()
            config.create_output_directories()
            
            # Step 2: Load all data
            self.logger.info("\n2. Loading input data...")
            if not self.loader.load_all_data():
                self.logger.error("Failed to load data. Exiting.")
                return False
            
            # Step 3: Get sample list
            all_samples = self.loader.get_all_sample_names()
            
            if self.test_mode:
                samples_to_process = all_samples[:5]  # Just 5 samples for testing
                self.logger.info(f"\n⚠ TEST MODE: Processing only {len(samples_to_process)} samples")
            elif self.n_samples:
                samples_to_process = all_samples[:self.n_samples]
                self.logger.info(f"\n⚠ Limited run: Processing {len(samples_to_process)} samples")
            else:
                samples_to_process = all_samples
                self.logger.info(f"\n✓ Full run: Processing all {len(samples_to_process)} samples")
            
            self.stats['total_samples'] = len(samples_to_process)
            
            # Step 4: Process all samples (parallelized)
            self.logger.info("\n3. Creating training dataset...")
            self.logger.info(f"   Using {config.N_CORES} CPU cores")
            self.logger.info(f"   Augmentation: {config.N_AUGMENTATIONS}x")
            self.logger.info(f"   Expected total files: ~{len(samples_to_process) * 51089 * config.N_AUGMENTATIONS:,}")
            
            self._process_all_samples(samples_to_process)
            
            # Step 5: Create metadata file
            self.logger.info("\n4. Creating metadata file...")
            self._create_metadata_file()
            
            # Step 6: Final validation
            self.logger.info("\n5. Running final validation...")
            self._validate_output()
            
            # Step 7: Print summary
            self.stats['end_time'] = datetime.now()
            self._print_summary()
            
            self.logger.info("\n✓ Step 1.4 completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"\n✗ ERROR: {str(e)}", exc_info=True)
            return False
    
    def _process_all_samples(self, samples: list):
        """
        Process all samples using multiprocessing.
        
        Args:
            samples: List of sample names to process
        """
        # Convert threemer_sequences DataFrame to dict for pickling
        # Keep index as keys, convert each row to dict
        threemer_sequences_dict = {}
        for idx in self.loader.threemer_sequences.index:
            threemer_sequences_dict[idx] = {
                '3mer_sequence': self.loader.threemer_sequences.loc[idx, '3mer_sequence']
            }
        
        # Get ordered list of region IDs
        region_ids_list = list(self.loader.threemer_sequences.index)
        
        sample_args = []
        for i, sample_name in enumerate(samples):
            sample_file_path = self.loader.sample_files[sample_name]
            sample_args.append((
                sample_name,
                sample_file_path,
                threemer_sequences_dict,
                region_ids_list,
                self.loader.tissue_mapping,
                i,
                len(samples)
            ))
        
        # Process samples in parallel
        with Pool(processes=config.N_CORES) as pool:
            results = list(tqdm(
                pool.imap(_process_sample_worker, sample_args),
                total=len(samples),
                desc="Processing samples"
            ))
        
        # Aggregate statistics
        for result in results:
            self.stats['total_regions'] += result['n_regions']
            self.stats['total_augmentations'] += result['n_augmentations']
            self.stats['total_files_created'] += result['n_files']
            self.stats['failed_regions'] += result['n_failed']
    
    def _process_single_sample(self, sample_name: str, sample_idx: int, total_samples: int):
        """
        Process a single sample: create augmented training files for all regions.
        
        Args:
            sample_name: Name of the sample
            sample_idx: Index of this sample
            total_samples: Total number of samples
            
        Returns:
            Dict: Processing statistics
        """
        stats = {
            'sample_name': sample_name,
            'n_regions': 0,
            'n_augmentations': 0,
            'n_files': 0,
            'n_failed': 0
        }
        
        try:
            # Load this sample's data on demand
            sample_data = self.loader.load_single_sample(sample_name)
            
            # Get sample info
            tissue = sample_data['tissue_label']
            tissue_one_hot = self.loader.get_tissue_one_hot(tissue)
            
            # Create 3-mer vocabulary
            threemer_to_id = create_threemer_vocab()
            
            sample_info = {
                'sample_name': sample_name,
                'tissue': tissue,
                'n_regions': len(sample_data['region_ids']),
                'n_reads': len(sample_data['read_ids'])
            }
            
            # Process each region in this sample
            n_regions = sample_info['n_regions']
            
            for region_idx in range(n_regions):
                try:
                    # Get region data
                    meth_reads, region_id, threemer_seq = self.loader.get_region_data(
                        sample_name, region_idx, sample_data
                    )
                    
                    # Create augmented versions
                    augmented_versions = self.augmenter.create_augmented_versions(
                        meth_reads, threemer_seq
                    )
                    
                    # Save each augmented version
                    for aug_reads, aug_threemer, aug_config in augmented_versions:
                        # Convert data to training format
                        training_data = convert_to_training_format(
                            aug_reads,
                            aug_threemer,
                            tissue_one_hot,
                            sample_info,
                            threemer_to_id
                        )
                        
                        # Save to file
                        filename = generate_filename(
                            sample_name,
                            region_id,
                            aug_config['version']
                        )
                        
                        save_training_file(training_data, filename)
                        
                        stats['n_files'] += 1
                        stats['n_augmentations'] += 1
                    
                    stats['n_regions'] += 1
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process region {region_idx} in {sample_name}: {str(e)}")
                    stats['n_failed'] += 1
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to process sample {sample_name}: {str(e)}")
            stats['n_failed'] = stats['n_regions']
            return stats
    
    def _create_metadata_file(self):
        """Create metadata CSV file summarizing all training data."""
        self.logger.info("   Scanning output directory...")
        
        metadata_entries = []
        
        # Scan all created files
        npz_files = [f for f in os.listdir(config.TRAIN_DIR) if f.endswith('.npz')]
        
        self.logger.info(f"   Found {len(npz_files)} training files")
        
        for npz_file in tqdm(npz_files, desc="Creating metadata"):
            try:
                # Parse filename: sample_name_augV.npz
                parts = npz_file.replace('.npz', '').rsplit('_aug', 1)
                sample_name = parts[0]
                aug_version = int(parts[1])
                
                # Load file to get metadata
                filepath = os.path.join(config.TRAIN_DIR, npz_file)
                data = np.load(filepath)
                
                tissue_idx = np.argmax(data['tissue_label'])
                tissue_name = str(data['tissue_name']) if 'tissue_name' in data else config.INDEX_TO_TISSUE.get(tissue_idx, 'Unknown')
                
                n_regions = data['dna_tokens'].shape[0]
                total_reads = int(data['n_reads'].sum()) if 'n_reads' in data else 0
                
                metadata_entries.append({
                    'filename': npz_file,
                    'sample_name': sample_name,
                    'tissue_type': tissue_name,
                    'tissue_index': tissue_idx,
                    'aug_version': aug_version,
                    'n_regions': n_regions,
                    'total_reads': total_reads,
                    'seq_length': config.MAX_SEQUENCE_LENGTH
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to process {npz_file}: {str(e)}")
        
        # Create DataFrame and save
        metadata_df = pd.DataFrame(metadata_entries)
        metadata_df.to_csv(config.METADATA_FILE, index=False)
        
        self.logger.info(f"   ✓ Metadata saved to {config.METADATA_FILE}")
        self.logger.info(f"   Total entries: {len(metadata_df)}")
    
    def _validate_output(self):
        """Validate created output files."""
        # Count files
        npz_files = [f for f in os.listdir(config.TRAIN_DIR) if f.endswith('.npz')]
        n_files = len(npz_files)
        
        self.logger.info(f"   Total files created: {n_files:,}")
        
        # Check a few random files
        self.logger.info("   Checking random sample files...")
        
        sample_files = np.random.choice(npz_files, size=min(10, len(npz_files)), replace=False)
        
        for npz_file in sample_files:
            filepath = os.path.join(config.TRAIN_DIR, npz_file)
            try:
                data = np.load(filepath)
                
                # Validate structure
                assert 'dna_tokens' in data, "Missing dna_tokens"
                assert 'methylation' in data, "Missing methylation"
                assert 'tissue_label' in data, "Missing tissue_label"
                
                # Validate shapes
                assert len(data['dna_tokens']) == config.MAX_SEQUENCE_LENGTH
                assert len(data['methylation']) == config.MAX_SEQUENCE_LENGTH
                assert len(data['tissue_label']) == config.N_TISSUES
                
                # Validate values
                assert data['dna_tokens'].dtype == np.int32
                assert data['methylation'].dtype == np.uint8
                assert data['tissue_label'].dtype == np.float32
                assert np.sum(data['tissue_label']) == 1.0  # One-hot
                
            except Exception as e:
                self.logger.error(f"   ✗ Validation failed for {npz_file}: {str(e)}")
                return False
        
        self.logger.info("   ✓ All sampled files passed validation")
        return True
    
    def _print_summary(self):
        """Print execution summary."""
        duration = self.stats['end_time'] - self.stats['start_time']
        
        print("\n" + "="*80)
        print("EXECUTION SUMMARY")
        print("="*80)
        print(f"Total samples processed: {self.stats['total_samples']}")
        print(f"Total regions processed: {self.stats['total_regions']:,}")
        print(f"Total augmentations created: {self.stats['total_augmentations']:,}")
        print(f"Total files created: {self.stats['total_files_created']:,}")
        print(f"Failed regions: {self.stats['failed_regions']}")
        print(f"\nExecution time: {duration}")
        print(f"Output directory: {config.OUTPUT_DIR}")
        print(f"Metadata file: {config.METADATA_FILE}")
        print("="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Create PyTorch training dataset for Step 1.4'
    )
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (process only 5 samples)')
    parser.add_argument('--n-samples', type=int, default=None,
                       help='Number of samples to process (default: all)')
    
    args = parser.parse_args()
    
    # Print configuration
    config.print_config_summary()
    
    # Create and run dataset creator
    creator = TrainingDatasetCreator(
        test_mode=args.test,
        n_samples=args.n_samples
    )
    
    success = creator.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
