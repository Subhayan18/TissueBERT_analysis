#!/usr/bin/env python3
"""
Stage 2 Blood-Masked Mixture Data Generator - ABSOLUTE LABELS
==============================================================

This dataloader generates synthetic mixtures with ABSOLUTE (not renormalized) labels.

Key differences from Phase 3:
- Input: Same mixed methylation (WITH blood component 60-100%)
- Output: Blood-masked proportions (21 tissues, ABSOLUTE values summing to 1-blood_prop)
- Training: Model learns true tissue proportions in blood-dominant cfDNA

Strategy:
1. Generate Phase 3-style mixtures (60-100% blood)
2. Remove blood from proportion labels
3. KEEP ABSOLUTE VALUES (do NOT renormalize to 1.0)
4. Model learns: "5% liver signal in 90% blood mixture" (not "50% liver")

Example:
  Mixture: 90% Blood + 5% Liver + 3% Pancreas + 2% Lung
  Labels: [0.05 Liver, 0.03 Pancreas, 0.02 Lung]  # Sums to 0.10, not 1.0!

This is learnable because labels match reality!

Author: Stage 2 Blood Deconvolution - Fixed
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


class BloodMaskedMixtureDataset(Dataset):
    """
    On-the-fly blood-masked mixture generation for Stage 2 training
    
    Generates realistic cfDNA mixtures (60-100% blood) but MASKS blood from labels.
    Forces model to predict non-blood tissue proportions in presence of blood.
    """

    def __init__(
        self,
        hdf5_path: str,
        metadata_csv: str,
        n_mixtures_per_epoch: int,
        pure_sample_ratio: float = 0.1,
        seed: int = None
    ):
        """
        Args:
            hdf5_path: Path to methylation_dataset.h5
            metadata_csv: Path to combined_metadata.csv
            n_mixtures_per_epoch: Number of mixtures per epoch
            pure_sample_ratio: Ratio of pure (non-blood) samples (default 0.1)
            seed: Random seed
        """
        self.hdf5_path = hdf5_path
        self.n_mixtures_per_epoch = n_mixtures_per_epoch
        self.pure_sample_ratio = pure_sample_ratio
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_csv)
        
        # Get ALL tissue information
        all_tissues = sorted(self.metadata['tissue_top_level'].unique())
        self.n_tissues_total = len(all_tissues)  # 22
        
        # Identify blood index
        self.blood_idx = all_tissues.index('Blood')
        
        # Create NON-BLOOD tissue list (21 tissues)
        self.unique_tissues = [t for t in all_tissues if t != 'Blood']
        self.n_tissues = len(self.unique_tissues)  # 21
        self.tissue_to_idx = {tissue: idx for idx, tissue in enumerate(self.unique_tissues)}
        
        # Store all tissues for mixing
        self.all_tissues = all_tissues
        self.all_tissue_to_idx = {tissue: idx for idx, tissue in enumerate(all_tissues)}
        
        # Open HDF5 to get dimensions
        with h5py.File(hdf5_path, 'r') as f:
            self.n_regions = f['methylation'].shape[1]
            self.seq_length = f['methylation'].shape[2]
        
        # Calculate pure vs mixed
        self.n_pure = int(n_mixtures_per_epoch * pure_sample_ratio)
        self.n_mixed = n_mixtures_per_epoch - self.n_pure
        
        # Organize samples
        self._organize_samples()
        
        print(f"\n  Initialized BloodMaskedMixtureDataset (Stage 2):")
        print(f"    Mixtures per epoch: {n_mixtures_per_epoch}")
        print(f"    Pure samples: {self.n_pure} ({pure_sample_ratio*100:.0f}%)")
        print(f"    Mixed samples: {self.n_mixed} ({(1-pure_sample_ratio)*100:.0f}%)")
        print(f"    Total tissues: {self.n_tissues_total} (including Blood)")
        print(f"    Output tissues: {self.n_tissues} (Blood MASKED)")
        print(f"    Regions: {self.n_regions}")

    def _organize_samples(self):
        """Organize samples by tissue and augmentation"""
        self.tissue_samples = {}
        
        for tissue in self.all_tissues:  # Include Blood for mixing
            self.tissue_samples[tissue] = {}
            
            for aug in range(5):
                samples = self.metadata[
                    (self.metadata['tissue_top_level'] == tissue) &
                    (self.metadata['aug_version'] == aug) &
                    (self.metadata['is_synthetic'] == False)
                ]
                
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
        Generate blood-masked mixture
        
        Returns:
            dict with:
                - methylation: [n_regions, seq_length] INCLUDES blood signal
                - proportions: [21] EXCLUDES blood, renormalized
                - is_pure: bool
        """
        is_pure = idx < self.n_pure
        
        if is_pure:
            return self._generate_pure_nonblood_sample()
        else:
            return self._generate_bloodmasked_mixture()

    def _load_region_means(self, sample_idx: int) -> np.ndarray:
        """Load and compute region means"""
        with h5py.File(self.hdf5_path, 'r') as f:
            methylation = f['methylation'][sample_idx]
        
        valid_mask = (methylation != 2).astype(float)
        region_sums = np.sum(methylation * valid_mask, axis=1)
        region_counts = np.sum(valid_mask, axis=1) + 1e-8
        region_means = region_sums / region_counts
        
        return region_means.astype(np.float32)

    def _generate_pure_nonblood_sample(self) -> Dict[str, torch.Tensor]:
        """Generate pure NON-BLOOD sample"""
        # Random non-blood tissue
        tissue = np.random.choice(self.unique_tissues)
        aug = np.random.randint(0, 5)
        
        # Get samples
        if aug not in self.tissue_samples[tissue] or len(self.tissue_samples[tissue][aug]) == 0:
            available_augs = [a for a in self.tissue_samples[tissue] if len(self.tissue_samples[tissue][a]) > 0]
            if len(available_augs) == 0:
                tissue = np.random.choice([t for t in self.unique_tissues if len(self.tissue_samples[t]) > 0])
                available_augs = [a for a in self.tissue_samples[tissue] if len(self.tissue_samples[tissue][a]) > 0]
            aug = np.random.choice(available_augs)
        
        sample_idx = np.random.choice(self.tissue_samples[tissue][aug])
        
        # Load
        with h5py.File(self.hdf5_path, 'r') as f:
            methylation = f['methylation'][sample_idx]
        
        # Create proportion vector (100% this tissue, NO blood dimension)
        proportions = np.zeros(self.n_tissues, dtype=np.float32)
        proportions[self.tissue_to_idx[tissue]] = 1.0
        
        return {
            'methylation': torch.tensor(methylation, dtype=torch.float32),
            'proportions': torch.tensor(proportions, dtype=torch.float32),
            'is_pure': True
        }

    def _generate_bloodmasked_mixture(self) -> Dict[str, torch.Tensor]:
        """
        Generate realistic cfDNA mixture with blood MASKED from labels
        
        Strategy:
        1. Generate 60-100% blood + 0-40% other tissues
        2. Mix methylation (INCLUDING blood signal)
        3. Create labels that EXCLUDE blood (only other tissues, renormalized)
        
        Model sees: Mixed signal with blood
        Model predicts: Non-blood tissue proportions
        """
        aug = np.random.randint(0, 5)
        
        # Blood proportion (60-100%)
        blood_prop = np.random.uniform(0.60, 1.0)
        
        # Number of other tissues (1-5)
        n_other = np.random.randint(1, 6)
        
        # Select other tissues (non-blood)
        selected_other = np.random.choice(
            self.unique_tissues, 
            size=min(n_other, len(self.unique_tissues)), 
            replace=False
        ).tolist()
        
        # Generate proportions for other tissues
        remaining_prop = 1.0 - blood_prop
        alpha = np.ones(len(selected_other)) * 0.5  # Skewed distribution
        other_proportions = (np.random.dirichlet(alpha) * remaining_prop).astype(np.float32)
        
        # === MIXING STEP (includes blood) ===
        # Start with blood
        blood_idx = np.random.choice(self.tissue_samples['Blood'][aug])
        mixed_means = blood_prop * self._load_region_means(blood_idx)
        
        # Add other tissues
        for tissue, prop in zip(selected_other, other_proportions):
            sample_idx = np.random.choice(self.tissue_samples[tissue][aug])
            means = self._load_region_means(sample_idx)
            mixed_means += prop * means
        
        # === LABEL CREATION (excludes blood) ===
        # Only non-blood tissues, renormalized to sum to 1.0
        proportions_masked = np.zeros(self.n_tissues, dtype=np.float32)
        
        for tissue, prop in zip(selected_other, other_proportions):
            proportions_masked[self.tissue_to_idx[tissue]] = prop
        
        # ABSOLUTE LABELS: Do NOT renormalize!
        # Labels should sum to (1 - blood_prop), not 1.0
        # This allows model to learn true absolute tissue proportions
        if proportions_masked.sum() == 0:
            # Edge case: 100% blood → tiny uniform signal
            proportions_masked = np.ones(self.n_tissues, dtype=np.float32) * 0.001
        
        # Expand to [n_regions, seq_length]
        mixed_methylation = np.repeat(mixed_means[:, np.newaxis], self.seq_length, axis=1)
        
        return {
            'methylation': torch.tensor(mixed_methylation, dtype=torch.float32),
            'proportions': torch.tensor(proportions_masked, dtype=torch.float32),
            'is_pure': False
        }


class BloodMaskedPreGeneratedDataset(Dataset):
    """
    Pre-generated blood-masked validation/test mixtures
    """

    def __init__(self, hdf5_path: str):
        """Load pre-generated blood-masked mixtures"""
        self.hdf5_path = hdf5_path
        
        with h5py.File(hdf5_path, 'r') as f:
            self.mixed_methylation = f['mixed_methylation'][:]
            self.true_proportions = f['true_proportions'][:]
            self.n_mixtures = f.attrs['n_mixtures']
            self.phase = f.attrs.get('phase', 'stage2')
            self.split = f.attrs.get('split', 'unknown')
        
        print(f"\n  Loaded BloodMaskedPreGeneratedDataset:")
        print(f"    File: {Path(hdf5_path).name}")
        print(f"    Phase: {self.phase}")
        print(f"    Split: {self.split}")
        print(f"    Samples: {self.n_mixtures}")
        print(f"    Output dim: {self.true_proportions.shape[1]} (blood-masked)")

    def __len__(self):
        return self.n_mixtures

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get pre-generated blood-masked mixture"""
        region_means = self.mixed_methylation[idx]
        proportions = self.true_proportions[idx]
        
        # Expand to [n_regions, seq_length]
        seq_length = 150
        methylation = np.repeat(region_means[:, np.newaxis], seq_length, axis=1)
        
        return {
            'methylation': torch.tensor(methylation, dtype=torch.float32),
            'proportions': torch.tensor(proportions, dtype=torch.float32)
        }


def create_bloodmasked_dataloaders(
    hdf5_path: str,
    metadata_csv: str,
    validation_h5: str,
    test_h5: str,
    batch_size: int,
    n_mixtures_per_epoch: int,
    pure_sample_ratio: float = 0.1,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for Stage 2 blood-masked training
    """
    print("\n" + "="*80)
    print("Creating Stage 2 Blood-Masked DataLoaders")
    print("="*80)
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    
    # Training dataset
    train_dataset = BloodMaskedMixtureDataset(
        hdf5_path=hdf5_path,
        metadata_csv=metadata_csv,
        n_mixtures_per_epoch=n_mixtures_per_epoch,
        pure_sample_ratio=pure_sample_ratio,
        seed=seed
    )
    
    # Validation/test datasets
    val_dataset = BloodMaskedPreGeneratedDataset(validation_h5)
    test_dataset = BloodMaskedPreGeneratedDataset(test_h5)
    
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
