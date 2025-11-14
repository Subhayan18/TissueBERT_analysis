"""
Data Loader Module for Step 1.4
Handles loading and validation of all input files.
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, Tuple, List
import config

class DataLoader:
    """
    Loads and validates all input data for training dataset creation.
    """
    
    def __init__(self):
        """Initialize the data loader."""
        self.beta_matrix = None
        self.synthetic_reads = {}
        self.threemer_sequences = None
        self.panel_regions = None
        self.tissue_mapping = None
        
    def load_all_data(self) -> bool:
        """
        Load all required input data (except synthetic reads - loaded on demand).
        
        Returns:
            bool: True if all data loaded successfully, False otherwise
        """
        print("\n" + "="*80)
        print("LOADING INPUT DATA")
        print("="*80 + "\n")
        
        try:
            # Load tissue mapping first
            print("1. Loading tissue label mapping...")
            self.tissue_mapping = self._load_tissue_mapping()
            print(f"   ✓ Loaded {len(self.tissue_mapping)} tissue types\n")
            
            # Load beta matrix
            print("2. Loading beta matrix...")
            self.beta_matrix = self._load_beta_matrix()
            print(f"   ✓ Shape: {self.beta_matrix.shape}")
            print(f"   ✓ Samples: {self.beta_matrix.shape[0]}")
            print(f"   ✓ Regions: {self.beta_matrix.shape[1]}\n")
            
            # Get list of synthetic read files (don't load them yet)
            print("3. Scanning synthetic reads directory...")
            self.sample_files = self._get_sample_files()
            print(f"   ✓ Found {len(self.sample_files)} sample files (will load on-demand)\n")
            
            # Load 3-mer sequences
            print("4. Loading 3-mer sequences...")
            self.threemer_sequences = self._load_threemer_sequences()
            print(f"   ✓ Loaded sequences for {len(self.threemer_sequences)} regions\n")
            
            # Load panel regions
            print("5. Loading panel BED file...")
            self.panel_regions = self._load_panel_regions()
            print(f"   ✓ Loaded {len(self.panel_regions)} regions\n")
            
            # Validate data consistency
            print("6. Validating data consistency...")
            if self._validate_data_consistency():
                print("   ✓ All data validated successfully!\n")
                return True
            else:
                print("   ✗ Data validation failed!\n")
                return False
                
        except Exception as e:
            print(f"\n✗ ERROR loading data: {str(e)}")
            return False
    
    def _load_tissue_mapping(self) -> Dict[str, int]:
        """Load tissue label to index mapping."""
        tissue_to_index, index_to_tissue = config.load_tissue_mapping()
        return tissue_to_index
    
    def _load_beta_matrix(self) -> pd.DataFrame:
        """
        Load the panel beta matrix from TSV file.
        First 5 columns are metadata: chr, start, end, startCpG, endCpG
        Remaining columns are beta values for each sample.
        
        Returns:
            pd.DataFrame: Beta matrix with metadata + beta values
        """
        beta_df = pd.read_csv(config.BETA_MATRIX_FILE, sep='\t')
        
        # Define metadata columns (first 5)
        metadata_cols = ['chr', 'start', 'end', 'startCpG', 'endCpG']
        
        # Extract beta value columns (everything except metadata)
        beta_value_cols = [col for col in beta_df.columns if col not in metadata_cols]
        
        # Validate beta values are in [0, 1] range (only check beta columns!)
        beta_values = beta_df[beta_value_cols]
        if beta_values.min().min() < 0 or beta_values.max().max() > 1:
            raise ValueError("Beta values must be between 0 and 1")
        
        print(f"   Metadata columns: {metadata_cols}")
        print(f"   Beta value columns: {len(beta_value_cols)} samples")
        
        # Return full dataframe (metadata + beta values) for consistency
        return beta_df
    
    def _get_sample_files(self) -> Dict[str, str]:
        """
        Get dictionary of sample names to file paths without loading data.
        
        Returns:
            Dict: {sample_name: file_path}
        """
        sample_files = {}
        
        npz_files = [f for f in os.listdir(config.SYNTHETIC_READS_DIR) 
                     if f.endswith('.npz')]
        
        for npz_file in npz_files:
            sample_name = npz_file.replace('.npz', '')
            file_path = os.path.join(config.SYNTHETIC_READS_DIR, npz_file)
            sample_files[sample_name] = file_path
        
        return sample_files
    
    def _load_synthetic_reads(self) -> Dict[str, Dict]:
        """
        Load all synthetic read files from the synthetic_reads directory.
        
        Returns:
            Dict: {sample_name: npz_data}
        """
        synthetic_reads = {}
        
        # Get all .npz files in synthetic_reads directory
        npz_files = [f for f in os.listdir(config.SYNTHETIC_READS_DIR) 
                     if f.endswith('.npz')]
        
        print(f"   Found {len(npz_files)} synthetic read files")
        
        for npz_file in npz_files:
            sample_name = npz_file.replace('.npz', '')
            file_path = os.path.join(config.SYNTHETIC_READS_DIR, npz_file)
            
            # Load npz file
            data = np.load(file_path, allow_pickle=True)
            
            synthetic_reads[sample_name] = {
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
        
        return synthetic_reads
    
    def load_single_sample(self, sample_name: str) -> Dict:
        """
        Load a single sample's synthetic reads on demand.
        
        Args:
            sample_name: Name of the sample to load
            
        Returns:
            Dict: Sample data dictionary
        """
        if sample_name not in self.sample_files:
            raise ValueError(f"Sample {sample_name} not found")
        
        file_path = self.sample_files[sample_name]
        data = np.load(file_path, allow_pickle=True)
        
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
        
        return sample_data
    
    def _load_threemer_sequences(self) -> pd.DataFrame:
        """
        Load 3-mer tokenized sequences.
        
        Returns:
            pd.DataFrame: 3-mer sequences with region_id as index
        """
        threemer_df = pd.read_csv(config.THREEMER_SEQUENCES_FILE, sep='\t')
        threemer_df.set_index('region_id', inplace=True)
        
        return threemer_df
    
    def _load_panel_regions(self) -> pd.DataFrame:
        """
        Load panel BED file with genomic coordinates.
        
        Returns:
            pd.DataFrame: Panel regions with columns [chrom, start, end, ...]
        """
        # BED file format: chrom, start, end, ...
        panel_df = pd.read_csv(config.PANEL_BED_FILE, sep='\t', 
                               header=None,
                               names=['chrom', 'start', 'end', 'name', 'score', 'strand'])
        
        # Create region_id from coordinates
        panel_df['region_id'] = (panel_df['chrom'] + '_' + 
                                 panel_df['start'].astype(str) + '_' + 
                                 panel_df['end'].astype(str))
        
        return panel_df
    
    def _validate_data_consistency(self) -> bool:
        """
        Validate that all loaded data is consistent.
        
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check that number of regions matches across datasets
            n_regions_beta = self.beta_matrix.shape[1] - 5  # Subtract metadata columns
            n_regions_3mer = len(self.threemer_sequences)
            n_regions_bed = len(self.panel_regions)
            
            print(f"   Regions in beta matrix: {n_regions_beta}")
            print(f"   Regions in 3-mer file: {n_regions_3mer}")
            print(f"   Regions in BED file: {n_regions_bed}")
            
            # Load one sample to check region count
            first_sample_name = list(self.sample_files.keys())[0]
            first_sample = self.load_single_sample(first_sample_name)
            n_regions_reads = len(first_sample['region_ids'])
            print(f"   Regions in synthetic reads: {n_regions_reads}")
            
            # Check if all match
            if not (n_regions_bed == n_regions_reads):
                print(f"   ✗ Region counts don't match!")
                return False
            
            # Check that tissue label is valid
            tissue = first_sample['tissue_label']
            if tissue not in self.tissue_mapping:
                print(f"   ✗ Unknown tissue label: {tissue}")
                return False
            
            return True
            
        except Exception as e:
            print(f"   ✗ Validation error: {str(e)}")
            return False
    
    def get_sample_info(self, sample_name: str) -> Dict:
        """
        Get information about a specific sample.
        
        Args:
            sample_name: Name of the sample
            
        Returns:
            Dict: Sample information
        """
        if sample_name not in self.sample_files:
            raise ValueError(f"Sample {sample_name} not found")
        
        sample_data = self.load_single_sample(sample_name)
        tissue = sample_data['tissue_label']
        
        return {
            'sample_name': sample_name,
            'tissue': tissue,
            'tissue_index': self.tissue_mapping[tissue],
            'n_regions': len(sample_data['region_ids']),
            'n_reads': len(sample_data['read_ids']),
            'reads_per_region': sample_data['reads_per_region'],
            'read_length': sample_data['read_length'],
            'biological_noise': sample_data['biological_noise']
        }
    
    def get_region_data(self, sample_name: str, region_idx: int, sample_data: Dict = None) -> Tuple[np.ndarray, str, str]:
        """
        Get methylation data and sequence for a specific region.
        
        Args:
            sample_name: Name of the sample
            region_idx: Index of the region (0-based)
            sample_data: Pre-loaded sample data (optional, loads if not provided)
            
        Returns:
            Tuple: (methylation_array, region_id, threemer_sequence)
        """
        if sample_data is None:
            sample_data = self.load_single_sample(sample_name)
        
        # Get region ID
        region_id = self.threemer_sequences.index[region_idx]
        
        # Find reads for this region
        region_mask = sample_data['region_indices'] == region_idx
        read_indices = np.where(region_mask)[0]
        
        # Extract methylation data for these reads
        methylation_reads = []
        for read_idx in read_indices:
            start = sample_data['read_boundaries'][read_idx]
            end = sample_data['read_boundaries'][read_idx + 1]
            meth_array = sample_data['methylation_data'][start:end]
            methylation_reads.append(meth_array)
        
        # Get 3-mer sequence
        threemer_seq = self.threemer_sequences.loc[region_id, '3mer_sequence']
        
        return methylation_reads, region_id, threemer_seq
    
    def get_all_sample_names(self) -> List[str]:
        """Get list of all sample names."""
        return list(self.sample_files.keys())
    
    def get_tissue_one_hot(self, tissue: str) -> np.ndarray:
        """
        Get one-hot encoded tissue label.
        
        Args:
            tissue: Tissue name
            
        Returns:
            np.ndarray: One-hot encoded vector of length N_TISSUES
        """
        tissue_idx = self.tissue_mapping[tissue]
        one_hot = np.zeros(config.N_TISSUES, dtype=np.float32)
        one_hot[tissue_idx] = 1.0
        return one_hot


def test_data_loader():
    """Test the data loader functionality."""
    print("\nTesting DataLoader...")
    
    loader = DataLoader()
    
    # Load all data
    if not loader.load_all_data():
        print("Failed to load data!")
        return
    
    # Test getting sample info
    sample_names = loader.get_all_sample_names()
    print(f"\nTotal samples: {len(sample_names)}")
    
    # Print info for first sample
    first_sample = sample_names[0]
    info = loader.get_sample_info(first_sample)
    print(f"\nFirst sample info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test getting region data
    meth_reads, region_id, threemer = loader.get_region_data(first_sample, 0)
    print(f"\nFirst region of first sample:")
    print(f"  Region ID: {region_id}")
    print(f"  Number of reads: {len(meth_reads)}")
    print(f"  First read length: {len(meth_reads[0])}")
    print(f"  3-mer sequence preview: {threemer[:100]}...")
    
    # Test tissue one-hot encoding
    tissue = info['tissue']
    one_hot = loader.get_tissue_one_hot(tissue)
    print(f"\nTissue one-hot encoding for '{tissue}':")
    print(f"  Vector length: {len(one_hot)}")
    print(f"  Non-zero index: {np.argmax(one_hot)}")
    
    print("\n✓ DataLoader test completed successfully!")


if __name__ == "__main__":
    test_data_loader()
