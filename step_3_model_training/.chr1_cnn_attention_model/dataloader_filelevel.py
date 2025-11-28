#!/usr/bin/env python3
"""
File-Level Dataset and DataLoader for Tissue Classification
============================================================

This dataloader returns ALL regions from each file as one sample,
matching what logistic regression did.

Key differences from region-level dataloader:
- Each sample = one ENTIRE FILE (all regions)
- Input shape: [batch_size, n_regions, seq_length]
- One label per file (not per region)
- Should enable 90%+ accuracy like logistic regression
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict


class FileLevel_MethylationDataset(Dataset):
    """
    File-level dataset - returns all regions from each file
    
    Unlike region-level sampling, this returns:
    - dna_tokens: [n_regions, seq_length]
    - methylation: [n_regions, seq_length]
    - tissue_label: scalar
    
    This matches logistic regression's approach.
    """

    def __init__(
        self,
        hdf5_path: str,
        file_indices: np.ndarray,
        mode: str = 'train'
    ):
        """
        Args:
            hdf5_path: Path to HDF5 file
            file_indices: Indices of files to use
            mode: 'train', 'val', or 'test'
        """
        self.hdf5_path = hdf5_path
        self.file_indices = file_indices
        self.mode = mode

        # Open HDF5 to get metadata
        with h5py.File(hdf5_path, 'r') as f:
            self.n_regions_per_file = f.attrs['n_regions_per_file']
            self.seq_length = f.attrs['seq_length']
            self.n_tissues = f.attrs['n_tissues']
            self.tissue_labels = f['tissue_labels'][file_indices]

        print(f"   Loaded {mode} dataset:")
        print(f"     - Files: {len(file_indices)}")
        print(f"     - Regions per file: {self.n_regions_per_file}")
        print(f"     - Tissues: {len(np.unique(self.tissue_labels))}")

    def __len__(self):
        return len(self.file_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get all regions from one file
        
        Returns:
            dict with:
                - dna_tokens: [n_regions, seq_length]
                - methylation: [n_regions, seq_length]
                - tissue_label: scalar
                - file_idx: int
        """
        file_idx = self.file_indices[idx]

        # Load ALL regions from this file
        with h5py.File(self.hdf5_path, 'r') as f:
            dna_tokens = f['dna_tokens'][file_idx]  # [n_regions, seq_length]
            methylation = f['methylation'][file_idx]  # [n_regions, seq_length]
            tissue_label = f['tissue_labels'][file_idx]  # scalar

        return {
            'dna_tokens': torch.tensor(dna_tokens, dtype=torch.long),
            'methylation': torch.tensor(methylation, dtype=torch.long),
            'tissue_label': torch.tensor(tissue_label, dtype=torch.long),
            'file_idx': file_idx
        }


def create_filelevel_dataloaders(
    hdf5_path: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create file-level dataloaders
    
    Args:
        hdf5_path: Path to HDF5 file
        train_csv, val_csv, test_csv: CSV files with file splits
        batch_size: Batch size (in FILES, not regions)
        num_workers: Number of workers
        pin_memory: Pin memory for GPU transfer
        
    Returns:
        train_loader, val_loader, test_loader
    """
    print("\n" + "="*80)
    print("Creating File-Level DataLoaders")
    print("="*80)

    # Load CSVs
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} files")
    print(f"  Val:   {len(val_df)} files")
    print(f"  Test:  {len(test_df)} files")

    # Load HDF5 metadata
    with h5py.File(hdf5_path, 'r') as f:
        hdf5_filenames = [fn.decode() for fn in f['metadata/filenames'][:]]

    filename_to_idx = {fn: idx for idx, fn in enumerate(hdf5_filenames)}

    # Get file indices
    train_indices = np.array([filename_to_idx[fn] for fn in train_df['filename']])
    val_indices = np.array([filename_to_idx[fn] for fn in val_df['filename']])
    test_indices = np.array([filename_to_idx[fn] for fn in test_df['filename']])

    # Sort indices
    train_indices = np.sort(train_indices)
    val_indices = np.sort(val_indices)
    test_indices = np.sort(test_indices)

    # Create datasets
    print(f"\nCreating datasets...")
    train_dataset = FileLevel_MethylationDataset(hdf5_path, train_indices, mode='train')
    val_dataset = FileLevel_MethylationDataset(hdf5_path, val_indices, mode='val')
    test_dataset = FileLevel_MethylationDataset(hdf5_path, test_indices, mode='test')

    # Create dataloaders
    print(f"\nCreating dataloaders...")
    print(f"  Batch size: {batch_size} files")
    print(f"  Num workers: {num_workers}")

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
        shuffle=False,
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

    print(f"\n✓ DataLoaders created successfully!")
    print(f"  Batches: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    """Test file-level dataloader"""
    
    HDF5_PATH = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/methylation_dataset_chr1.h5"
    TRAIN_CSV = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/chr1_subset/train_files.csv"
    VAL_CSV = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/chr1_subset/val_files.csv"
    TEST_CSV = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/chr1_subset/test_files.csv"
    
    train_loader, val_loader, test_loader = create_filelevel_dataloaders(
        hdf5_path=HDF5_PATH,
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        test_csv=TEST_CSV,
        batch_size=4,
        num_workers=0
    )
    
    print("\n" + "="*80)
    print("Testing Dataloader")
    print("="*80)
    
    # Get one batch
    batch = next(iter(train_loader))
    
    print(f"\nBatch contents:")
    print(f"  dna_tokens: {batch['dna_tokens'].shape}")
    print(f"  methylation: {batch['methylation'].shape}")
    print(f"  tissue_label: {batch['tissue_label'].shape}")
    print(f"  Labels: {batch['tissue_label']}")
    
    print("\n✓ File-level dataloader test passed!")
