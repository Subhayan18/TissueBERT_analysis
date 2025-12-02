#!/usr/bin/env python3
"""
On-the-Fly Mixture Data Generator for Tissue Deconvolution Training
=====================================================================

This dataloader generates synthetic mixtures during training for maximum
diversity, while using pre-generated validation/test sets for reproducibility.

Key features:
- Generates mixtures on-the-fly (never see same mixture twice)
- Phase-specific mixing strategies (1: 2-tissue, 2: 3-5 tissue, 3: cfDNA-like)
- Loads pure samples from HDF5
- Returns mixed methylation + ground truth proportions
- Memory efficient (computes region means on-the-fly)

Author: Mixture Deconvolution Project
Date: December 2024
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
import random


class MixtureGeneratorDataset(Dataset):
    """
    On-the-fly mixture generation dataset for training
    
    Generates synthetic mixtures by:
    1. Sampling N tissues (based on phase)
    2. Generating proportions (based on phase strategy)
    3. Loading pure samples from HDF5
    4. Computing region means
    5. Linearly mixing: mixed = Σ(proportion_i × sample_i_means)
    
    Each __getitem__ call generates a NEW mixture.
    """

    def __init__(
        self,
        hdf5_path: str,
        metadata_csv: str,
        phase: int,
        n_mixtures_per_epoch: int,
        pure_sample_ratio: float = 0.2,
        seed: int = None
    ):
        """
        Args:
            hdf5_path: Path to methylation_dataset.h5
            metadata_csv: Path to combined_metadata.csv
            phase: 1 (2-tissue), 2 (3-5 tissue), or 3 (cfDNA-like)
            n_mixtures_per_epoch: Number of mixtures to generate per epoch
            pure_sample_ratio: Ratio of pure samples to include (default 0.2 = 20%)
            seed: Random seed for reproducibility (optional)
        """
        self.hdf5_path = hdf5_path
        self.phase = phase
        self.n_mixtures_per_epoch = n_mixtures_per_epoch
        self.pure_sample_ratio = pure_sample_ratio
        
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_csv)
        
        # Get tissue information
        self.unique_tissues = sorted(self.metadata['tissue_top_level'].unique())
        self.n_tissues = len(self.unique_tissues)
        self.tissue_to_idx = {tissue: idx for idx, tissue in enumerate(self.unique_tissues)}
        
        # Open HDF5 once to get dimensions
        with h5py.File(hdf5_path, 'r') as f:
            self.n_regions = f['methylation'].shape[1]
            self.seq_length = f['methylation'].shape[2]
        
        # Calculate number of pure vs mixed samples
        self.n_pure = int(n_mixtures_per_epoch * pure_sample_ratio)
        self.n_mixed = n_mixtures_per_epoch - self.n_pure
        
        # Group samples by tissue and augmentation version
        self._organize_samples()
        
        print(f"\n  Initialized MixtureGeneratorDataset:")
        print(f"    Phase: {phase}")
        print(f"    Mixtures per epoch: {n_mixtures_per_epoch}")
        print(f"    Pure samples: {self.n_pure} ({pure_sample_ratio*100:.0f}%)")
        print(f"    Mixed samples: {self.n_mixed} ({(1-pure_sample_ratio)*100:.0f}%)")
        print(f"    Tissues: {self.n_tissues}")
        print(f"    Regions: {self.n_regions}")

    def _organize_samples(self):
        """Organize samples by tissue and augmentation version for fast lookup"""
        self.tissue_samples = {}
        
        for tissue in self.unique_tissues:
            self.tissue_samples[tissue] = {}
            
            for aug in range(5):  # aug0 through aug4
                # Prefer real samples
                samples = self.metadata[
                    (self.metadata['tissue_top_level'] == tissue) &
                    (self.metadata['aug_version'] == aug) &
                    (self.metadata['is_synthetic'] == False)
                ]
                
                # If no real samples, allow synthetic
                if len(samples) == 0:
                    samples = self.metadata[
                        (self.metadata['tissue_top_level'] == tissue) &
                        (self.metadata['aug_version'] == aug)
                    ]
                
                if len(samples) > 0:
                    self.tissue_samples[tissue][aug] = samples.index.tolist()

    def __len__(self):
        return self.n_mixtures_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Generate one mixture sample
        
        Returns:
            dict with:
                - methylation: [n_regions, seq_length] mixed methylation pattern
                - proportions: [n_tissues] ground truth tissue proportions
                - is_pure: bool (True if pure sample, False if mixture)
        """
        # Decide if this should be a pure sample or mixture
        is_pure = idx < self.n_pure
        
        if is_pure:
            return self._generate_pure_sample()
        else:
            if self.phase == 1:
                return self._generate_phase1_mixture()
            elif self.phase == 2:
                return self._generate_phase2_mixture()
            elif self.phase == 3:
                return self._generate_phase3_mixture()
            else:
                raise ValueError(f"Invalid phase: {self.phase}")

    def _load_region_means(self, sample_idx: int) -> np.ndarray:
        """
        Load methylation and compute region means for one sample
        
        Args:
            sample_idx: Index in HDF5 file
            
        Returns:
            region_means: [n_regions] array
        """
        with h5py.File(self.hdf5_path, 'r') as f:
            methylation = f['methylation'][sample_idx]  # [n_regions, seq_length]
        
        # Compute region means (handle missing values = 2)
        valid_mask = (methylation != 2).astype(float)
        region_sums = np.sum(methylation * valid_mask, axis=1)
        region_counts = np.sum(valid_mask, axis=1) + 1e-8
        region_means = region_sums / region_counts
        
        return region_means.astype(np.float32)

    def _generate_pure_sample(self) -> Dict[str, torch.Tensor]:
        """Generate a pure (single-tissue) sample"""
        # Random tissue and augmentation
        tissue = np.random.choice(self.unique_tissues)
        aug = np.random.randint(0, 5)
        
        # Get samples for this tissue+aug
        if aug not in self.tissue_samples[tissue] or len(self.tissue_samples[tissue][aug]) == 0:
            # Fallback to any available augmentation
            available_augs = [a for a in self.tissue_samples[tissue] if len(self.tissue_samples[tissue][a]) > 0]
            if len(available_augs) == 0:
                # Last resort: pick any tissue
                tissue = np.random.choice([t for t in self.unique_tissues if len(self.tissue_samples[t]) > 0])
                available_augs = [a for a in self.tissue_samples[tissue] if len(self.tissue_samples[tissue][a]) > 0]
            aug = np.random.choice(available_augs)
        
        # Sample one file
        sample_idx = np.random.choice(self.tissue_samples[tissue][aug])
        
        # Load region means
        region_means = self._load_region_means(sample_idx)
        
        # Create proportion vector (100% this tissue)
        proportions = np.zeros(self.n_tissues, dtype=np.float32)
        proportions[self.tissue_to_idx[tissue]] = 1.0
        
        # For compatibility with model, expand to [n_regions, seq_length]
        # by repeating the region means
        with h5py.File(self.hdf5_path, 'r') as f:
            methylation = f['methylation'][sample_idx]
        
        return {
            'methylation': torch.tensor(methylation, dtype=torch.float32),
            'proportions': torch.tensor(proportions, dtype=torch.float32),
            'is_pure': True
        }

    def _generate_phase1_mixture(self) -> Dict[str, torch.Tensor]:
        """
        Generate 2-tissue mixture
        
        Strategy:
        - Select 2 different tissues
        - Use predefined proportion splits
        - Same augmentation version
        """
        # Sample augmentation version
        aug = np.random.randint(0, 5)
        
        # Sample 2 different tissues
        attempts = 0
        while attempts < 100:
            tissues = np.random.choice(self.unique_tissues, size=2, replace=False)
            tissue_a, tissue_b = tissues[0], tissues[1]
            
            # Check if both tissues have samples for this augmentation
            has_a = aug in self.tissue_samples[tissue_a] and len(self.tissue_samples[tissue_a][aug]) > 0
            has_b = aug in self.tissue_samples[tissue_b] and len(self.tissue_samples[tissue_b][aug]) > 0
            
            if has_a and has_b:
                break
            
            attempts += 1
        
        if attempts >= 100:
            # Fallback: use any available augmentation
            aug = np.random.randint(0, 5)
            available_tissues = [t for t in self.unique_tissues 
                                if aug in self.tissue_samples[t] and len(self.tissue_samples[t][aug]) > 0]
            if len(available_tissues) < 2:
                # Last resort: use different augmentations
                return self._generate_phase1_mixture()
            tissue_a, tissue_b = np.random.choice(available_tissues, size=2, replace=False)
        
        # Sample proportion strategy
        proportion_pairs = [
            (0.5, 0.5),
            (0.6, 0.4), (0.4, 0.6),
            (0.7, 0.3), (0.3, 0.7),
            (0.8, 0.2), (0.2, 0.8)
        ]
        prop_a, prop_b = proportion_pairs[np.random.randint(len(proportion_pairs))]
        
        # Sample files
        sample_a_idx = np.random.choice(self.tissue_samples[tissue_a][aug])
        sample_b_idx = np.random.choice(self.tissue_samples[tissue_b][aug])
        
        # Load region means
        means_a = self._load_region_means(sample_a_idx)
        means_b = self._load_region_means(sample_b_idx)
        
        # Linear mixing
        mixed_means = prop_a * means_a + prop_b * means_b
        
        # Create proportion vector
        proportions = np.zeros(self.n_tissues, dtype=np.float32)
        proportions[self.tissue_to_idx[tissue_a]] = prop_a
        proportions[self.tissue_to_idx[tissue_b]] = prop_b
        
        # Expand to [n_regions, seq_length] by repeating
        mixed_methylation = np.repeat(mixed_means[:, np.newaxis], self.seq_length, axis=1)
        
        return {
            'methylation': torch.tensor(mixed_methylation, dtype=torch.float32),
            'proportions': torch.tensor(proportions, dtype=torch.float32),
            'is_pure': False
        }

    def _generate_phase2_mixture(self) -> Dict[str, torch.Tensor]:
        """
        Generate 3-5 tissue mixture
        
        Strategy:
        - Select 3-5 different tissues
        - Use Dirichlet distribution (alpha=2.0)
        - Same augmentation version
        """
        # Number of tissues
        n_components = np.random.randint(3, 6)  # 3, 4, or 5
        
        # Sample augmentation version
        aug = np.random.randint(0, 5)
        
        # Sample tissues
        attempts = 0
        while attempts < 100:
            selected_tissues = np.random.choice(self.unique_tissues, size=n_components, replace=False)
            
            # Check all have samples for this augmentation
            all_valid = all(
                aug in self.tissue_samples[t] and len(self.tissue_samples[t][aug]) > 0
                for t in selected_tissues
            )
            
            if all_valid:
                break
            
            attempts += 1
        
        if attempts >= 100:
            # Fallback: use any available tissues with this augmentation
            available_tissues = [t for t in self.unique_tissues 
                                if aug in self.tissue_samples[t] and len(self.tissue_samples[t][aug]) > 0]
            if len(available_tissues) < n_components:
                n_components = len(available_tissues)
            selected_tissues = np.random.choice(available_tissues, size=n_components, replace=False)
        
        # Generate proportions using Dirichlet
        alpha = np.ones(n_components) * 2.0
        proportions_components = np.random.dirichlet(alpha).astype(np.float32)
        
        # Mix
        mixed_means = np.zeros(self.n_regions, dtype=np.float32)
        
        for tissue, prop in zip(selected_tissues, proportions_components):
            sample_idx = np.random.choice(self.tissue_samples[tissue][aug])
            means = self._load_region_means(sample_idx)
            mixed_means += prop * means
        
        # Create proportion vector
        proportions = np.zeros(self.n_tissues, dtype=np.float32)
        for tissue, prop in zip(selected_tissues, proportions_components):
            proportions[self.tissue_to_idx[tissue]] = prop
        
        # Expand to [n_regions, seq_length]
        mixed_methylation = np.repeat(mixed_means[:, np.newaxis], self.seq_length, axis=1)
        
        return {
            'methylation': torch.tensor(mixed_methylation, dtype=torch.float32),
            'proportions': torch.tensor(proportions, dtype=torch.float32),
            'is_pure': False
        }

    def _generate_phase3_mixture(self) -> Dict[str, torch.Tensor]:
        """
        Generate realistic cfDNA-like mixture
        
        Strategy:
        - Blood always present (60-90%)
        - 2-6 other tissues
        - Skewed distribution (Dirichlet alpha=0.5)
        """
        # Check Blood is available
        if 'Blood' not in self.unique_tissues:
            # Fallback to phase 2 if Blood not available
            return self._generate_phase2_mixture()
        
        # Sample augmentation version
        aug = np.random.randint(0, 5)
        
        # Check Blood has samples
        if aug not in self.tissue_samples['Blood'] or len(self.tissue_samples['Blood'][aug]) == 0:
            # Try another augmentation
            available_augs = [a for a in self.tissue_samples['Blood'] if len(self.tissue_samples['Blood'][a]) > 0]
            if len(available_augs) == 0:
                return self._generate_phase2_mixture()
            aug = np.random.choice(available_augs)
        
        # Blood proportion (60-90%)
        blood_prop = np.random.uniform(0.6, 0.9)
        
        # Number of other tissues (2-6)
        non_blood_tissues = [t for t in self.unique_tissues if t != 'Blood']
        n_other = np.random.randint(2, 7)
        n_other = min(n_other, len(non_blood_tissues))
        
        # Sample other tissues
        attempts = 0
        while attempts < 100:
            selected_other = np.random.choice(non_blood_tissues, size=n_other, replace=False)
            
            # Check all have samples
            all_valid = all(
                aug in self.tissue_samples[t] and len(self.tissue_samples[t][aug]) > 0
                for t in selected_other
            )
            
            if all_valid:
                break
            
            attempts += 1
        
        if attempts >= 100:
            # Fallback
            available = [t for t in non_blood_tissues 
                        if aug in self.tissue_samples[t] and len(self.tissue_samples[t][aug]) > 0]
            if len(available) < n_other:
                n_other = len(available)
            selected_other = np.random.choice(available, size=n_other, replace=False)
        
        # Generate proportions for other tissues (skewed)
        remaining_prop = 1.0 - blood_prop
        alpha = np.ones(n_other) * 0.5  # Skewed distribution
        other_proportions = (np.random.dirichlet(alpha) * remaining_prop).astype(np.float32)
        
        # Mix: Start with Blood
        blood_idx = np.random.choice(self.tissue_samples['Blood'][aug])
        mixed_means = blood_prop * self._load_region_means(blood_idx)
        
        # Add other tissues
        for tissue, prop in zip(selected_other, other_proportions):
            sample_idx = np.random.choice(self.tissue_samples[tissue][aug])
            means = self._load_region_means(sample_idx)
            mixed_means += prop * means
        
        # Create proportion vector
        proportions = np.zeros(self.n_tissues, dtype=np.float32)
        proportions[self.tissue_to_idx['Blood']] = blood_prop
        for tissue, prop in zip(selected_other, other_proportions):
            proportions[self.tissue_to_idx[tissue]] = prop
        
        # Expand to [n_regions, seq_length]
        mixed_methylation = np.repeat(mixed_means[:, np.newaxis], self.seq_length, axis=1)
        
        return {
            'methylation': torch.tensor(mixed_methylation, dtype=torch.float32),
            'proportions': torch.tensor(proportions, dtype=torch.float32),
            'is_pure': False
        }


class PreGeneratedMixtureDataset(Dataset):
    """
    Dataset for pre-generated validation/test mixtures
    
    Loads from HDF5 files created by generate_mixture_datasets.py
    """

    def __init__(self, hdf5_path: str):
        """
        Args:
            hdf5_path: Path to phase*_validation/test_mixtures.h5
        """
        self.hdf5_path = hdf5_path
        
        # Load all data into memory (small enough)
        with h5py.File(hdf5_path, 'r') as f:
            self.mixed_methylation = f['mixed_methylation'][:]
            self.true_proportions = f['true_proportions'][:]
            self.n_mixtures = f.attrs['n_mixtures']
            self.phase = f.attrs['phase']
            self.split = f.attrs['split']
        
        print(f"\n  Loaded PreGeneratedMixtureDataset:")
        print(f"    File: {Path(hdf5_path).name}")
        print(f"    Phase: {self.phase}")
        print(f"    Split: {self.split}")
        print(f"    Samples: {self.n_mixtures}")

    def __len__(self):
        return self.n_mixtures

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get one pre-generated mixture
        
        Returns:
            dict with:
                - methylation: [n_regions, seq_length] 
                - proportions: [n_tissues]
        """
        # The pre-generated data already has region means, but we need
        # to expand to [n_regions, seq_length] for model compatibility
        region_means = self.mixed_methylation[idx]  # [n_regions]
        proportions = self.true_proportions[idx]  # [n_tissues]
        
        # Expand to [n_regions, seq_length] by repeating
        seq_length = 150  # Fixed
        methylation = np.repeat(region_means[:, np.newaxis], seq_length, axis=1)
        
        return {
            'methylation': torch.tensor(methylation, dtype=torch.float32),
            'proportions': torch.tensor(proportions, dtype=torch.float32)
        }


def create_mixture_dataloaders(
    hdf5_path: str,
    metadata_csv: str,
    validation_h5: str,
    test_h5: str,
    phase: int,
    batch_size: int,
    n_mixtures_per_epoch: int,
    pure_sample_ratio: float = 0.2,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for mixture training
    
    Args:
        hdf5_path: Path to methylation_dataset.h5 (pure samples)
        metadata_csv: Path to combined_metadata.csv
        validation_h5: Path to phase*_validation_mixtures.h5
        test_h5: Path to phase*_test_mixtures.h5
        phase: 1, 2, or 3
        batch_size: Batch size
        n_mixtures_per_epoch: Number of training mixtures per epoch
        pure_sample_ratio: Ratio of pure samples in training
        num_workers: Number of DataLoader workers
        seed: Random seed
        
    Returns:
        train_loader, val_loader, test_loader
    """
    print("\n" + "="*80)
    print("Creating Mixture Deconvolution DataLoaders")
    print("="*80)
    print(f"Phase: {phase}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    
    # Training dataset (on-the-fly generation)
    train_dataset = MixtureGeneratorDataset(
        hdf5_path=hdf5_path,
        metadata_csv=metadata_csv,
        phase=phase,
        n_mixtures_per_epoch=n_mixtures_per_epoch,
        pure_sample_ratio=pure_sample_ratio,
        seed=seed
    )
    
    # Validation dataset (pre-generated)
    val_dataset = PreGeneratedMixtureDataset(validation_h5)
    
    # Test dataset (pre-generated)
    test_dataset = PreGeneratedMixtureDataset(test_h5)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    
    print(f"\n✓ DataLoaders created successfully!")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    """Test mixture generator"""
    
    HDF5_PATH = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/methylation_dataset.h5"
    METADATA_CSV = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/combined_metadata.csv"
    VAL_H5 = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/mixture_data/phase1_validation_mixtures.h5"
    TEST_H5 = "/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/mixture_data/phase1_test_mixtures.h5"
    
    print("Testing Mixture Generator...")
    
    train_loader, val_loader, test_loader = create_mixture_dataloaders(
        hdf5_path=HDF5_PATH,
        metadata_csv=METADATA_CSV,
        validation_h5=VAL_H5,
        test_h5=TEST_H5,
        phase=1,
        batch_size=4,
        n_mixtures_per_epoch=100,
        pure_sample_ratio=0.2,
        num_workers=0,
        seed=42
    )
    
    print("\n" + "="*80)
    print("Testing DataLoader")
    print("="*80)
    
    # Test training batch
    print("\nTraining batch:")
    batch = next(iter(train_loader))
    print(f"  Methylation: {batch['methylation'].shape}")
    print(f"  Proportions: {batch['proportions'].shape}")
    print(f"  Proportions sum: {batch['proportions'].sum(dim=1)}")
    print(f"  Is pure: {batch.get('is_pure', 'N/A')}")
    
    # Test validation batch
    print("\nValidation batch:")
    batch = next(iter(val_loader))
    print(f"  Methylation: {batch['methylation'].shape}")
    print(f"  Proportions: {batch['proportions'].shape}")
    print(f"  Proportions sum: {batch['proportions'].sum(dim=1)}")
    
    print("\n✓ Mixture generator test passed!")
