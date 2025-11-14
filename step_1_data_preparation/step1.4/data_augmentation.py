"""
Data Augmentation Module for Step 1.4
Implements jittering, coverage subsampling, and strand flipping.
"""

import numpy as np
from typing import List, Tuple
import config

class DataAugmenter:
    """
    Handles all data augmentation strategies for training dataset creation.
    """
    
    def __init__(self, random_seed: int = None):
        """
        Initialize the augmenter.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed if random_seed else config.RANDOM_SEED
        self.rng = np.random.default_rng(self.random_seed)
    
    def augment_region(self, 
                       methylation_reads: List[np.ndarray],
                       threemer_sequence: str,
                       aug_config: dict) -> Tuple[List[np.ndarray], str]:
        """
        Apply augmentation to a region's data.
        
        Args:
            methylation_reads: List of methylation arrays for reads
            threemer_sequence: Space-separated 3-mer sequence
            aug_config: Augmentation configuration dict
            
        Returns:
            Tuple: (augmented_methylation_reads, augmented_threemer_sequence)
        """
        # Make copies to avoid modifying originals
        aug_meth_reads = [read.copy() for read in methylation_reads]
        aug_threemer = threemer_sequence
        
        # Apply jittering if requested
        if aug_config['jitter']:
            aug_meth_reads = self._apply_jitter(
                aug_meth_reads, 
                jitter_percent=aug_config['jitter_percent']
            )
        
        # Apply coverage subsampling
        if aug_config['coverage'] < len(aug_meth_reads):
            aug_meth_reads = self._subsample_reads(
                aug_meth_reads,
                target_coverage=aug_config['coverage']
            )
        
        # Apply strand flipping if requested
        if aug_config['strand_flip']:
            aug_meth_reads, aug_threemer = self._flip_strand(
                aug_meth_reads,
                aug_threemer
            )
        
        return aug_meth_reads, aug_threemer
    
    def _apply_jitter(self, 
                      methylation_reads: List[np.ndarray],
                      jitter_percent: float = 0.05) -> List[np.ndarray]:
        """
        Apply ±jitter_percent noise to methylation values.
        
        For each CpG site, randomly adjust methylation probability by ±jitter_percent.
        This simulates biological variation and measurement noise.
        
        Args:
            methylation_reads: List of methylation arrays
            jitter_percent: Amount of jitter (e.g., 0.05 for ±5%)
            
        Returns:
            List of jittered methylation arrays
        """
        jittered_reads = []
        
        for read in methylation_reads:
            jittered_read = read.copy()
            
            # Only jitter CpG positions (0 or 1, not 2)
            cpg_mask = read < 2
            
            if not np.any(cpg_mask):
                jittered_reads.append(jittered_read)
                continue
            
            # Calculate jitter for each CpG
            # We'll work with probabilities, not binary values
            # Convert binary methylation to probability
            meth_prob = read[cpg_mask].astype(float)
            
            # Add random noise: uniform between -jitter_percent and +jitter_percent
            noise = self.rng.uniform(-jitter_percent, jitter_percent, size=meth_prob.shape)
            jittered_prob = np.clip(meth_prob + noise, 0.0, 1.0)
            
            # Convert back to binary using the jittered probability
            jittered_binary = (self.rng.random(size=jittered_prob.shape) < jittered_prob).astype(np.uint8)
            
            jittered_read[cpg_mask] = jittered_binary
            jittered_reads.append(jittered_read)
        
        return jittered_reads
    
    def _subsample_reads(self,
                        methylation_reads: List[np.ndarray],
                        target_coverage: int) -> List[np.ndarray]:
        """
        Randomly subsample reads to achieve target coverage.
        
        Args:
            methylation_reads: List of methylation arrays
            target_coverage: Desired number of reads
            
        Returns:
            List of subsampled methylation arrays
        """
        n_reads = len(methylation_reads)
        
        if n_reads <= target_coverage:
            # If we have fewer reads than target, return all
            return methylation_reads
        
        # Randomly select target_coverage reads
        selected_indices = self.rng.choice(n_reads, size=target_coverage, replace=False)
        selected_indices = np.sort(selected_indices)  # Keep in order
        
        subsampled_reads = [methylation_reads[i] for i in selected_indices]
        
        return subsampled_reads
    
    def _flip_strand(self,
                    methylation_reads: List[np.ndarray],
                    threemer_sequence: str) -> Tuple[List[np.ndarray], str]:
        """
        Flip reads to reverse complement strand.
        
        Args:
            methylation_reads: List of methylation arrays
            threemer_sequence: Space-separated 3-mer sequence
            
        Returns:
            Tuple: (flipped_reads, flipped_threemer_sequence)
        """
        # Reverse the order of reads (5' to 3' becomes 3' to 5')
        flipped_reads = [read[::-1] for read in methylation_reads]
        
        # Reverse complement the 3-mer sequence
        flipped_threemer = self._reverse_complement_3mers(threemer_sequence)
        
        return flipped_reads, flipped_threemer
    
    def _reverse_complement_3mers(self, threemer_sequence: str) -> str:
        """
        Convert 3-mer sequence to reverse complement.
        
        Args:
            threemer_sequence: Space-separated 3-mer sequence
            
        Returns:
            str: Reverse complemented 3-mer sequence
        """
        # Complement mapping
        complement = {
            'A': 'T', 'T': 'A',
            'C': 'G', 'G': 'C',
            'N': 'N'
        }
        
        # Split into 3-mers
        threemers = threemer_sequence.split()
        
        # Reverse complement each 3-mer
        rc_threemers = []
        for kmer in threemers:
            rc_kmer = ''.join([complement.get(base, 'N') for base in reversed(kmer)])
            rc_threemers.append(rc_kmer)
        
        # Reverse the order of 3-mers
        rc_threemers = rc_threemers[::-1]
        
        return ' '.join(rc_threemers)
    
    def create_augmented_versions(self,
                                 methylation_reads: List[np.ndarray],
                                 threemer_sequence: str) -> List[Tuple[List[np.ndarray], str, dict]]:
        """
        Create all augmented versions according to config.AUGMENTATION_CONFIG.
        
        Args:
            methylation_reads: List of methylation arrays for reads
            threemer_sequence: Space-separated 3-mer sequence
            
        Returns:
            List of tuples: [(aug_meth_reads, aug_threemer, aug_config), ...]
        """
        augmented_versions = []
        
        for aug_config in config.AUGMENTATION_CONFIG:
            aug_meth_reads, aug_threemer = self.augment_region(
                methylation_reads,
                threemer_sequence,
                aug_config
            )
            
            augmented_versions.append((aug_meth_reads, aug_threemer, aug_config))
        
        return augmented_versions
    
    def validate_augmentation(self,
                            original_reads: List[np.ndarray],
                            augmented_reads: List[np.ndarray],
                            aug_config: dict) -> bool:
        """
        Validate that augmentation was applied correctly.
        
        Args:
            original_reads: Original methylation reads
            augmented_reads: Augmented methylation reads
            aug_config: Augmentation configuration
            
        Returns:
            bool: True if validation passes
        """
        # Check coverage subsampling
        if aug_config['coverage'] < len(original_reads):
            if len(augmented_reads) != aug_config['coverage']:
                print(f"  ✗ Coverage validation failed: expected {aug_config['coverage']}, "
                      f"got {len(augmented_reads)}")
                return False
        
        # Check that all reads have valid values (0, 1, or 2)
        for i, read in enumerate(augmented_reads):
            if not np.all((read == 0) | (read == 1) | (read == 2)):
                print(f"  ✗ Read {i} has invalid methylation values")
                return False
        
        return True


def test_augmenter():
    """Test the data augmentation functionality."""
    print("\nTesting DataAugmenter...")
    
    augmenter = DataAugmenter(random_seed=42)
    
    # Create synthetic test data
    print("\n1. Creating test data...")
    n_reads = 1000
    read_length = 150
    
    # Create reads with 70% methylation at CpGs
    test_reads = []
    for i in range(n_reads):
        read = np.random.choice([0, 1, 2], size=read_length, p=[0.15, 0.25, 0.60])
        test_reads.append(read)
    
    test_threemer = "ATG TGC GCA CAT ATG TGC GCA CAT ATG TGC"  # Example sequence
    
    print(f"   Created {len(test_reads)} reads of length {read_length}")
    
    # Test jittering
    print("\n2. Testing jittering...")
    jitter_config = {'jitter': True, 'jitter_percent': 0.05, 
                     'coverage': 1000, 'strand_flip': False}
    jittered_reads, _ = augmenter.augment_region(test_reads, test_threemer, jitter_config)
    
    # Calculate difference in methylation
    orig_meth = np.concatenate([r[r < 2] for r in test_reads])
    jit_meth = np.concatenate([r[r < 2] for r in jittered_reads])
    diff = np.abs(orig_meth.mean() - jit_meth.mean())
    print(f"   Original methylation: {orig_meth.mean():.3f}")
    print(f"   Jittered methylation: {jit_meth.mean():.3f}")
    print(f"   Difference: {diff:.3f}")
    
    # Test subsampling
    print("\n3. Testing coverage subsampling...")
    for target_cov in [500, 100, 50, 30, 10]:
        subsample_config = {'jitter': False, 'jitter_percent': 0.0,
                           'coverage': target_cov, 'strand_flip': False}
        sub_reads, _ = augmenter.augment_region(test_reads, test_threemer, subsample_config)
        print(f"   Target: {target_cov}x, Got: {len(sub_reads)} reads")
    
    # Test strand flipping
    print("\n4. Testing strand flipping...")
    flip_config = {'jitter': False, 'jitter_percent': 0.0,
                  'coverage': 1000, 'strand_flip': True}
    flipped_reads, flipped_threemer = augmenter.augment_region(
        test_reads, test_threemer, flip_config
    )
    print(f"   Original 3-mers: {test_threemer}")
    print(f"   Flipped 3-mers:  {flipped_threemer}")
    print(f"   Original first read (first 20): {test_reads[0][:20]}")
    print(f"   Flipped first read (first 20):  {flipped_reads[0][:20]}")
    
    # Test creating all augmented versions
    print("\n5. Testing full augmentation pipeline...")
    all_versions = augmenter.create_augmented_versions(test_reads, test_threemer)
    print(f"   Created {len(all_versions)} augmented versions:")
    for i, (aug_reads, aug_threemer, aug_config) in enumerate(all_versions):
        print(f"   v{i} ({aug_config['description']}): {len(aug_reads)} reads")
    
    # Test validation
    print("\n6. Testing validation...")
    for aug_reads, aug_threemer, aug_config in all_versions:
        valid = augmenter.validate_augmentation(test_reads, aug_reads, aug_config)
        status = "✓" if valid else "✗"
        print(f"   {status} v{aug_config['version']} validation")
    
    print("\n✓ DataAugmenter test completed successfully!")


if __name__ == "__main__":
    test_augmenter()
