#!/usr/bin/env python3
"""
PyTorch Dataset and DataLoader for Methylation Deconvolution.
Includes tissue-balanced sampling, CpG dropout augmentation, and validation wrapper.

Usage:
    from dataset_dataloader import create_dataloaders, validate_dataloaders
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        hdf5_path='path/to/data.h5',
        batch_size=32,
        num_workers=4,
        cpg_dropout_rate=0.05
    )
    
    # Validate with small subset
    validate_dataloaders(train_loader, val_loader, n_samples=10)
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
import time
from typing import Tuple, Optional, Dict, List
import warnings


class TissueBalancedBatchSampler(Sampler):
    """
    Custom batch sampler for tissue-balanced sampling without exceeding 2^24 limit.
    
    Samples files according to tissue weights, then randomly samples regions within files.
    This avoids the torch.multinomial 2^24 limit (WeightedRandomSampler fails with 23M regions).
    """
    
    def __init__(self, dataset, batch_size: int, num_batches: int):
        """
        Args:
            dataset: MethylationDataset with file_weights attribute
            batch_size: Number of regions per batch
            num_batches: Number of batches per epoch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches
        
        # Get file-level weights
        if hasattr(dataset, 'file_weights'):
            self.file_weights = dataset.file_weights
        else:
            # No balancing - uniform
            n_files = len(dataset.file_indices)
            self.file_weights = np.ones(n_files) / n_files
        
        self.n_regions_per_file = dataset.n_regions_per_file
        self.file_indices = dataset.file_indices
        
    def __iter__(self):
        """Generate batches by sampling files then regions."""
        for _ in range(self.num_batches):
            batch_indices = []
            
            for _ in range(self.batch_size):
                # Sample a file according to tissue weights
                file_pos = np.random.choice(len(self.file_weights), p=self.file_weights)
                
                # Sample a random region from this file
                region_idx = np.random.randint(0, self.n_regions_per_file)
                
                # Convert to global region index
                global_idx = file_pos * self.n_regions_per_file + region_idx
                batch_indices.append(global_idx)
            
            yield batch_indices
    
    def __len__(self):
        return self.num_batches


class MethylationDataset(Dataset):
    """
    PyTorch Dataset for methylation deconvolution from HDF5 file.
    
    Supports:
    - Region-level sampling
    - Tissue-balanced sampling
    - CpG dropout augmentation
    - Efficient HDF5 random access
    """
    
    def __init__(
        self,
        hdf5_path: str,
        file_indices: np.ndarray,
        mode: str = 'train',
        cpg_dropout_rate: float = 0.0,
        augment_coverage: bool = True,
        tissue_balanced: bool = False
    ):
        """
        Args:
            hdf5_path: Path to HDF5 file
            file_indices: Indices of files to use from HDF5
            mode: 'train', 'val', or 'test'
            cpg_dropout_rate: Probability of converting CpG (0/1) to missing (2)
            augment_coverage: If True, use all augmentation versions
            tissue_balanced: If True, enable tissue-balanced sampling
        """
        self.hdf5_path = hdf5_path
        self.file_indices = file_indices
        self.mode = mode
        self.cpg_dropout_rate = cpg_dropout_rate
        self.augment_coverage = augment_coverage
        self.tissue_balanced = tissue_balanced
        
        # Open HDF5 file to get metadata
        with h5py.File(hdf5_path, 'r') as f:
            self.n_regions_per_file = f.attrs['n_regions_per_file']
            self.seq_length = f.attrs['seq_length']
            self.n_tissues = f.attrs['n_tissues']
            
            # Get tissue labels for the selected files
            self.tissue_labels = f['tissue_labels'][file_indices]
            
            # Get augmentation versions
            aug_versions = f['metadata/aug_versions'][file_indices]
            self.aug_versions = aug_versions
        
        # Calculate total number of regions
        self.total_regions = len(file_indices) * self.n_regions_per_file
        
        # Create region-to-file mapping
        self.region_to_file = np.repeat(file_indices, self.n_regions_per_file)
        self.region_to_region_idx = np.tile(np.arange(self.n_regions_per_file), len(file_indices))
        
        # For tissue-balanced sampling, compute weights
        if tissue_balanced:
            self._compute_tissue_weights()
        
        print(f"   Loaded {mode} dataset:")
        print(f"     - Files: {len(file_indices)}")
        print(f"     - Total regions: {self.total_regions:,}")
        print(f"     - Tissues: {len(np.unique(self.tissue_labels))}")
        print(f"     - CpG dropout: {cpg_dropout_rate}")
        print(f"     - Tissue balanced: {tissue_balanced}")
    
    def _compute_tissue_weights(self):
        """
        Compute sampling weights for tissue balancing at FILE level (not region level).
        
        Problem: Region-level weights (23M) exceed torch.multinomial's 2^24 limit.
        Solution: Weight files (455) by tissue, then sample regions uniformly within files.
        """
        # Count FILES per tissue (not regions!)
        tissue_counts = Counter(self.tissue_labels)
        n_tissues = len(tissue_counts)
        
        # Compute weight for each FILE
        # Weight = 1 / (n_files_in_tissue * n_tissues)
        self.file_weights = np.zeros(len(self.file_indices))
        
        for file_pos, tissue in enumerate(self.tissue_labels):
            weight = 1.0 / (tissue_counts[tissue] * n_tissues)
            self.file_weights[file_pos] = weight
        
        # Normalize
        self.file_weights /= self.file_weights.sum()
        
        # Don't create region-level weights - would exceed 2^24 limit!
        # Instead, we'll use custom batch sampler
        
        print(f"     - Tissue distribution (FILE-level balancing):")
        for tissue, count in sorted(tissue_counts.items()):
            weight = 1.0 / (count * n_tissues)
            print(f"       Tissue {tissue}: {count} files, weight={weight:.6f}")
    
    def get_sampler(self):
        """
        Get sampler for tissue balancing.
        
        Returns None - we use a custom batch sampler instead to avoid 2^24 limit.
        The batch sampler is created in create_dataloaders().
        """
        return None
    
    def __len__(self):
        return self.total_regions
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single region.
        
        Returns:
            dict with keys:
                - dna_tokens: [seq_length] uint8
                - methylation: [seq_length] uint8 (0=unmeth, 1=meth, 2=no_cpg)
                - n_reads: int32
                - tissue_label: uint8
                - file_idx: int (for debugging)
                - region_idx: int (for debugging)
        """
        # Get file and region indices
        file_idx = self.region_to_file[idx]
        region_idx = self.region_to_region_idx[idx]
        
        # Load data from HDF5
        with h5py.File(self.hdf5_path, 'r') as f:
            dna_tokens = f['dna_tokens'][file_idx, region_idx]
            methylation = f['methylation'][file_idx, region_idx].copy()  # Copy for modification
            n_reads = f['n_reads'][file_idx, region_idx]
            tissue_label = f['tissue_labels'][file_idx]
        
        # Apply CpG dropout augmentation (only in training mode)
        if self.mode == 'train' and self.cpg_dropout_rate > 0:
            methylation = self._apply_cpg_dropout(methylation)
        
        return {
            'dna_tokens': torch.tensor(dna_tokens, dtype=torch.long),
            'methylation': torch.tensor(methylation, dtype=torch.long),
            'n_reads': torch.tensor(n_reads, dtype=torch.long),
            'tissue_label': torch.tensor(tissue_label, dtype=torch.long),
            'file_idx': file_idx,
            'region_idx': region_idx
        }
    
    def _apply_cpg_dropout(self, methylation: np.ndarray) -> np.ndarray:
        """
        Randomly convert some CpG sites (0 or 1) to missing (2).
        
        Args:
            methylation: Array with values 0, 1, or 2
        
        Returns:
            Modified methylation array
        """
        # Find positions with actual CpG data (not already missing)
        cpg_positions = methylation < 2
        n_cpg = cpg_positions.sum()
        
        if n_cpg == 0:
            return methylation
        
        # Randomly select positions to dropout
        n_dropout = int(n_cpg * self.cpg_dropout_rate)
        if n_dropout == 0:
            return methylation
        
        # Get indices of CpG positions
        cpg_indices = np.where(cpg_positions)[0]
        dropout_indices = np.random.choice(cpg_indices, size=n_dropout, replace=False)
        
        # Set selected positions to missing (2)
        methylation[dropout_indices] = 2
        
        return methylation


def create_dataloaders(
    hdf5_path: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    batch_size: int = 32,
    num_workers: int = 4,
    cpg_dropout_rate: float = 0.05,
    tissue_balanced: bool = True,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        hdf5_path: Path to HDF5 file
        train_csv: Path to train split CSV
        val_csv: Path to validation split CSV
        test_csv: Path to test split CSV
        batch_size: Batch size
        num_workers: Number of data loading workers
        cpg_dropout_rate: CpG dropout rate for training augmentation
        tissue_balanced: Use tissue-balanced sampling for training
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("\n" + "="*80)
    print("Creating DataLoaders")
    print("="*80)
    
    # Load split CSVs
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} files")
    print(f"  Val:   {len(val_df)} files")
    print(f"  Test:  {len(test_df)} files")
    
    # Load HDF5 metadata to map filenames to indices
    with h5py.File(hdf5_path, 'r') as f:
        hdf5_filenames = [fn.decode() for fn in f['metadata/filenames'][:]]
    
    filename_to_idx = {fn: idx for idx, fn in enumerate(hdf5_filenames)}
    
    # Get indices for each split
    train_indices = np.array([filename_to_idx[fn] for fn in train_df['filename']])
    val_indices = np.array([filename_to_idx[fn] for fn in val_df['filename']])
    test_indices = np.array([filename_to_idx[fn] for fn in test_df['filename']])
    
    # Sort indices (h5py requires indices to be in increasing order)
    train_indices = np.sort(train_indices)
    val_indices = np.sort(val_indices)
    test_indices = np.sort(test_indices)
    
    # Create datasets
    print(f"\nCreating datasets...")
    train_dataset = MethylationDataset(
        hdf5_path=hdf5_path,
        file_indices=train_indices,
        mode='train',
        cpg_dropout_rate=cpg_dropout_rate,
        tissue_balanced=tissue_balanced
    )
    
    val_dataset = MethylationDataset(
        hdf5_path=hdf5_path,
        file_indices=val_indices,
        mode='val',
        cpg_dropout_rate=0.0,  # No augmentation for validation
        tissue_balanced=False
    )
    
    test_dataset = MethylationDataset(
        hdf5_path=hdf5_path,
        file_indices=test_indices,
        mode='test',
        cpg_dropout_rate=0.0,  # No augmentation for test
        tissue_balanced=False
    )
    
    # Create dataloaders
    print(f"\nCreating dataloaders...")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    
    # For training with tissue balancing, use custom batch sampler
    if tissue_balanced:
        # Calculate number of batches per epoch
        num_train_batches = len(train_dataset) // batch_size
        
        batch_sampler = TissueBalancedBatchSampler(
            train_dataset,
            batch_size=batch_size,
            num_batches=num_train_batches
        )
        
        print(f"  Using custom batch sampler (avoids 2^24 limit)")
        print(f"  Batches per epoch: {num_train_batches}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0)
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0)
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Sequential for validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    
    print("\n✓ DataLoaders created successfully!")
    
    return train_loader, val_loader, test_loader


def validate_dataloaders(
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_samples: int = 10,
    device: str = 'cuda'
) -> Dict:
    """
    Validate dataloaders with a small subset and produce benchmark report.
    
    Args:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        n_samples: Number of batches to test
        device: Device to test on
    
    Returns:
        dict with validation results and benchmarks
    """
    print("\n" + "="*80)
    print(f"DataLoader Validation (testing {n_samples} batches)")
    print("="*80)
    
    results = {
        'train': {},
        'val': {},
        'errors': [],
        'warnings': []
    }
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Test training loader
    print(f"\n1. Testing Training DataLoader...")
    print("-" * 80)
    
    try:
        train_stats = _validate_single_loader(
            train_loader, n_samples, device, mode='train'
        )
        results['train'] = train_stats
    except Exception as e:
        results['errors'].append(f"Train loader error: {str(e)}")
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test validation loader
    print(f"\n2. Testing Validation DataLoader...")
    print("-" * 80)
    
    try:
        val_stats = _validate_single_loader(
            val_loader, n_samples, device, mode='val'
        )
        results['val'] = val_stats
    except Exception as e:
        results['errors'].append(f"Val loader error: {str(e)}")
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print(f"\n3. Validation Summary")
    print("=" * 80)
    
    if results['errors']:
        print(f"\n✗ Errors encountered:")
        for error in results['errors']:
            print(f"   - {error}")
    else:
        print(f"\n✓ All validation checks passed!")
    
    if results['warnings']:
        print(f"\n⚠ Warnings:")
        for warning in results['warnings']:
            print(f"   - {warning}")
    
    # Comparison
    if results['train'] and results['val']:
        print(f"\n4. Train vs Val Comparison")
        print("-" * 80)
        print(f"   Loading speed:")
        print(f"     Train: {results['train']['samples_per_sec']:.1f} samples/sec")
        print(f"     Val:   {results['val']['samples_per_sec']:.1f} samples/sec")
        print(f"   Memory usage:")
        print(f"     Train: {results['train']['memory_per_batch_mb']:.1f} MB/batch")
        print(f"     Val:   {results['val']['memory_per_batch_mb']:.1f} MB/batch")
    
    print("\n" + "="*80)
    
    return results


def _validate_single_loader(
    loader: DataLoader,
    n_samples: int,
    device: torch.device,
    mode: str
) -> Dict:
    """Validate a single dataloader."""
    
    stats = {
        'n_batches_tested': 0,
        'total_samples': 0,
        'shapes': defaultdict(list),
        'dtypes': {},
        'value_ranges': defaultdict(dict),
        'timing': [],
        'memory_usage': [],
        'tissue_distribution': Counter(),
        'methylation_stats': defaultdict(list)
    }
    
    print(f"   Testing {n_samples} batches...")
    
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= n_samples:
            break
        
        start_time = time.time()
        
        # Move to device
        batch_gpu = {k: v.to(device) if torch.is_tensor(v) else v 
                     for k, v in batch.items()}
        
        elapsed = time.time() - start_time
        stats['timing'].append(elapsed)
        
        # Memory usage
        if device.type == 'cuda':
            memory_mb = torch.cuda.memory_allocated() / 1e6
            stats['memory_usage'].append(memory_mb)
        
        # Validate shapes and dtypes
        for key, value in batch.items():
            if torch.is_tensor(value):
                stats['shapes'][key].append(value.shape)
                stats['dtypes'][key] = value.dtype
                
                # Check for NaN/Inf
                if torch.is_floating_point(value):
                    if torch.any(torch.isnan(value)):
                        print(f"      ✗ NaN values in {key}")
                    if torch.any(torch.isinf(value)):
                        print(f"      ✗ Inf values in {key}")
                
                # Value ranges
                if key in ['dna_tokens', 'methylation', 'tissue_label']:
                    min_val = value.min().item()
                    max_val = value.max().item()
                    stats['value_ranges'][key][batch_idx] = (min_val, max_val)
        
        # Tissue distribution
        tissue_labels = batch['tissue_label'].cpu().numpy()
        stats['tissue_distribution'].update(tissue_labels)
        
        # Methylation statistics
        methylation = batch['methylation'].cpu().numpy()
        meth_rate = (methylation == 1).sum() / (methylation < 2).sum()
        missing_rate = (methylation == 2).sum() / methylation.size
        stats['methylation_stats']['meth_rate'].append(meth_rate)
        stats['methylation_stats']['missing_rate'].append(missing_rate)
        
        stats['n_batches_tested'] += 1
        stats['total_samples'] += batch['dna_tokens'].shape[0]
        
        # Progress
        if (batch_idx + 1) % max(1, n_samples // 5) == 0:
            print(f"      Batch {batch_idx+1}/{n_samples}: "
                  f"{elapsed*1000:.1f}ms, "
                  f"{memory_mb if device.type == 'cuda' else 0:.0f}MB")
    
    # Compute summary statistics
    stats['samples_per_sec'] = stats['total_samples'] / sum(stats['timing'])
    stats['mean_batch_time_ms'] = np.mean(stats['timing']) * 1000
    stats['std_batch_time_ms'] = np.std(stats['timing']) * 1000
    
    if stats['memory_usage']:
        stats['memory_per_batch_mb'] = np.mean(stats['memory_usage'])
    else:
        stats['memory_per_batch_mb'] = 0
    
    # Print summary
    print(f"\n   Results:")
    print(f"     ✓ Batches tested: {stats['n_batches_tested']}")
    print(f"     ✓ Total samples: {stats['total_samples']}")
    print(f"     ✓ Loading speed: {stats['samples_per_sec']:.1f} samples/sec")
    print(f"     ✓ Batch time: {stats['mean_batch_time_ms']:.1f} ± {stats['std_batch_time_ms']:.1f} ms")
    
    if stats['memory_per_batch_mb'] > 0:
        print(f"     ✓ Memory per batch: {stats['memory_per_batch_mb']:.1f} MB")
    
    # Print shapes
    print(f"\n   Shapes (first batch):")
    for key, shapes in stats['shapes'].items():
        print(f"     {key}: {shapes[0]} (dtype={stats['dtypes'][key]})")
    
    # Print value ranges
    print(f"\n   Value ranges:")
    for key in ['dna_tokens', 'methylation', 'tissue_label']:
        if key in stats['value_ranges']:
            all_mins = [v[0] for v in stats['value_ranges'][key].values()]
            all_maxs = [v[1] for v in stats['value_ranges'][key].values()]
            print(f"     {key}: [{min(all_mins)}, {max(all_maxs)}]")
    
    # Print tissue distribution
    print(f"\n   Tissue distribution (top 10):")
    for tissue, count in stats['tissue_distribution'].most_common(10):
        print(f"     Tissue {tissue}: {count} samples")
    
    # Print methylation stats
    print(f"\n   Methylation statistics:")
    mean_meth = np.mean(stats['methylation_stats']['meth_rate'])
    mean_missing = np.mean(stats['methylation_stats']['missing_rate'])
    print(f"     Mean methylation rate: {mean_meth*100:.2f}%")
    print(f"     Mean missing rate: {mean_missing*100:.2f}%")
    
    return stats


# Example usage
if __name__ == "__main__":
    """
    Example: Validate dataloaders with 10 samples.
    """
    
    # Paths
    HDF5_PATH = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/methylation_dataset.h5"
    SPLITS_DIR = Path("/home/chattopa/data_storage/MethAtlas_WGBSanalysis/data_splits")
    
    TRAIN_CSV = SPLITS_DIR / "train_files.csv"
    VAL_CSV = SPLITS_DIR / "val_files.csv"
    TEST_CSV = SPLITS_DIR / "test_files.csv"
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        hdf5_path=HDF5_PATH,
        train_csv=str(TRAIN_CSV),
        val_csv=str(VAL_CSV),
        test_csv=str(TEST_CSV),
        batch_size=32,
        num_workers=4,
        cpg_dropout_rate=0.05,
        tissue_balanced=True,
        pin_memory=True
    )
    
    # Validate with 10 samples
    results = validate_dataloaders(
        train_loader=train_loader,
        val_loader=val_loader,
        n_samples=10,
        device='cuda'
    )
    
    print("\n✓ Validation complete!")
