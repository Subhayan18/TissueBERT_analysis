# Mixture Deconvolution Implementation Plan
## Comprehensive Step-by-Step Guide for Training Tissue Proportion Prediction Model

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture Modifications](#architecture-modifications)
3. [Synthetic Mixture Generation Strategy](#synthetic-mixture-generation-strategy)
4. [Training Pipeline Design](#training-pipeline-design)
5. [Evaluation Framework](#evaluation-framework)
6. [Implementation Timeline](#implementation-timeline)
7. [File Structure](#file-structure)
8. [Detailed Step-by-Step Execution](#detailed-step-by-step-execution)
9. [Quality Control Checkpoints](#quality-control-checkpoints)
10. [Troubleshooting Guide](#troubleshooting-guide)

---

## Project Overview

### Current State
- **Model**: File-level MLP with mean methylation aggregation
- **Performance**: 97.8% validation accuracy on single-tissue classification
- **Architecture**: 51,089 regions → 512 hidden → MLP → 22 tissue classes
- **Training Data**: 765 samples (119 unique × 5 augmentations)
- **Data Format**: HDF5 with methylation patterns per region

### Target State
- **Model**: Same architecture with modified output layer
- **Task**: Predict tissue proportions in mixed samples
- **Output**: 22-dimensional probability distribution (proportions sum to 1.0)
- **Training**: Progressive complexity (2-tissue → 5+ tissues → realistic distributions)

### Key Design Decisions
- ✅ Fine-tune existing model (leverage learned features)
- ✅ Sigmoid + normalization output (flexible learning)
- ✅ MSE loss on proportions (interpretable)
- ✅ Linear mixing of region means (weighted averages)
- ✅ On-the-fly training generation, pre-generated validation/test
- ✅ 80% mixtures + 20% pure samples in training
- ✅ Global random seed: 42

---

## Architecture Modifications

### Current Output Layer
```python
Current Architecture (Single-Tissue Classification):
├── MLP layers output: [batch, 1024]
├── Final linear: 1024 → 22
└── CrossEntropyLoss (implicit softmax)

Output interpretation: argmax → single tissue class
```

### New Output Layer
```python
Modified Architecture (Mixture Deconvolution):
├── MLP layers output: [batch, 1024]
├── Final linear: 1024 → 22
├── Sigmoid activation: [batch, 22] (independent probabilities)
├── L1 normalization: proportions.sum(dim=1) = 1.0
└── MSE Loss on normalized proportions

Output interpretation: 22-dimensional proportion vector
```

### Implementation Details

**Step 1: Modify Model Class**
```python
class TissueBERTDeconvolution(nn.Module):
    """
    Modified from TissueBERT for mixture deconvolution.
    Changes:
    1. Output activation: Softmax → Sigmoid + L1 normalization
    2. Loss function: CrossEntropy → MSE on proportions
    """
    
    def __init__(self, n_regions=51089, hidden_size=512, num_classes=22, dropout=0.1):
        super().__init__()
        
        # Keep existing layers from trained model
        self.projection = nn.Linear(n_regions, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(1024, num_classes)
        
        # NEW: Sigmoid activation for independent tissue probabilities
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, methylation):
        """
        Args:
            methylation: [batch, n_regions, 150] - methylation patterns
        
        Returns:
            proportions: [batch, 22] - normalized tissue proportions (sum to 1.0)
        """
        # Aggregate methylation per region (same as before)
        valid_mask = (methylation != 2).float()
        region_means = (methylation * valid_mask).sum(dim=2) / (valid_mask.sum(dim=2) + 1e-8)
        # Shape: [batch, n_regions]
        
        # Feature extraction (same as before)
        features = self.projection(region_means)  # [batch, 512]
        hidden = self.mlp(features)  # [batch, 1024]
        logits = self.classifier(hidden)  # [batch, 22]
        
        # NEW: Sigmoid + L1 normalization
        sigmoid_outputs = self.sigmoid(logits)  # [batch, 22]
        proportions = sigmoid_outputs / sigmoid_outputs.sum(dim=1, keepdim=True)  # Normalize to sum=1
        
        return proportions
```

**Step 2: Load Pre-trained Weights**
```python
def load_pretrained_model(checkpoint_path, device='cuda'):
    """
    Load pre-trained single-tissue model and adapt for mixture deconvolution.
    
    Args:
        checkpoint_path: Path to checkpoint_best_acc.pt (97.8% accuracy model)
        device: 'cuda' or 'cpu'
    
    Returns:
        model: TissueBERTDeconvolution with loaded weights
    """
    # Initialize new model
    model = TissueBERTDeconvolution(
        n_regions=51089,
        hidden_size=512,
        num_classes=22,
        dropout=0.1
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Load all weights (architecture is identical except forward pass)
    model.load_state_dict(state_dict, strict=True)
    
    print(f"✓ Loaded pre-trained model from {checkpoint_path}")
    print(f"  Original validation accuracy: {checkpoint.get('val_accuracy', 'N/A')}")
    
    return model.to(device)
```

**Step 3: Define Loss Function**
```python
def mixture_mse_loss(predicted_proportions, true_proportions):
    """
    Mean Squared Error on tissue proportions.
    
    Args:
        predicted_proportions: [batch, 22] - model predictions (sum to 1.0)
        true_proportions: [batch, 22] - ground truth mixing weights (sum to 1.0)
    
    Returns:
        loss: scalar MSE value
    """
    # Ensure both sum to 1 (sanity check)
    assert torch.allclose(predicted_proportions.sum(dim=1), torch.ones(predicted_proportions.size(0), device=predicted_proportions.device), atol=1e-5)
    assert torch.allclose(true_proportions.sum(dim=1), torch.ones(true_proportions.size(0), device=true_proportions.device), atol=1e-5)
    
    # MSE across all tissues
    loss = nn.functional.mse_loss(predicted_proportions, true_proportions)
    
    return loss
```

---

## Synthetic Mixture Generation Strategy

### Overview
Generate synthetic mixtures by linearly combining methylation patterns from pure tissue samples with known proportions.

**Linear Mixing Formula:**
```
For N tissues with proportions [p₁, p₂, ..., pₙ] where Σpᵢ = 1:

Mixed_methylation = Σ (pᵢ × sample_i_region_means)

Where:
- sample_i_region_means: [51089] vector of mean methylation per region
- Result: [51089] vector representing mixed sample
```

### Phase 1: 2-Tissue Mixtures

**Objective**: Teach model to decompose simple binary mixtures

**Configuration:**
```yaml
Phase 1 - Two Tissue Mixtures:
  Tissue Combinations:
    - Total possible pairs: C(22,2) = 231 unique tissue pairs
    - Use ALL pairs for comprehensive coverage
    - Critical: Ensure tissues in pair are DIFFERENT types
    
  Proportion Strategies:
    Strategy A (Equiproportional):
      - 50% Tissue A + 50% Tissue B
    
    Strategy B (Imbalanced):
      - 60% Tissue A + 40% Tissue B
      - 40% Tissue A + 60% Tissue B (symmetric)
      - 70% Tissue A + 30% Tissue B
      - 30% Tissue A + 70% Tissue B (symmetric)
      - 80% Tissue A + 20% Tissue B
      - 20% Tissue A + 80% Tissue B (symmetric)
    
    Total per pair: 7 proportion variants
    Total training examples: 231 pairs × 7 variants = 1,617 base mixtures
  
  Augmentation Handling:
    - Mix samples from SAME augmentation version (consistent coverage)
    - Train on ALL 5 augmentation versions (aug0 through aug4)
    - This creates: 1,617 × 5 = 8,085 unique 2-tissue mixtures per epoch
    
  Training Data (80% mixtures + 20% pure):
    - Mixtures: Generate 2,000 per epoch (on-the-fly, random sampling)
    - Pure samples: Include 500 pure samples per epoch (100-0-0-...-0 proportions)
    - Total per epoch: 2,500 training examples
    
  Validation Data (pre-generated, fixed):
    - 500 mixtures (fixed combinations, saved to HDF5)
    - Seed: 42 for reproducibility
    
  Test Data (pre-generated, fixed):
    - 500 mixtures (different from validation, saved to HDF5)
    - Seed: 43 for reproducibility
```

**Critical Constraint: Ensure Different Tissues**
```python
def validate_tissue_pair(sample_a_tissue, sample_b_tissue):
    """
    CRITICAL: Ensure two samples are from DIFFERENT tissue types.
    
    With replicates, augmentations, and synthetic samples of same tissue,
    we MUST verify tissue types differ to create true mixtures.
    
    Args:
        sample_a_tissue: Tissue label of sample A (0-21)
        sample_b_tissue: Tissue label of sample B (0-21)
    
    Returns:
        bool: True if tissues differ, False otherwise
    """
    return sample_a_tissue != sample_b_tissue

# Usage in mixture generation
while True:
    sample_a_idx = random.randint(0, n_samples-1)
    sample_b_idx = random.randint(0, n_samples-1)
    
    tissue_a = get_tissue_label(sample_a_idx)
    tissue_b = get_tissue_label(sample_b_idx)
    
    if validate_tissue_pair(tissue_a, tissue_b):
        # Valid pair - create mixture
        break
    # else: retry sampling
```

### Phase 2: 3-5 Tissue Mixtures

**Objective**: Increase complexity to moderate multi-tissue scenarios

**Configuration:**
```yaml
Phase 2 - Multi-Tissue Mixtures (3-5 tissues):
  Tissue Combinations:
    - Random selection of 3-5 different tissues per mixture
    - Ensure all selected tissues are DIFFERENT types (critical!)
    - Sample uniformly across all 22 tissue types
    
  Proportion Strategies:
    Strategy A (Equiproportional):
      - 3 tissues: 33.3% - 33.3% - 33.4%
      - 4 tissues: 25% - 25% - 25% - 25%
      - 5 tissues: 20% - 20% - 20% - 20% - 20%
    
    Strategy B (Random Dirichlet):
      - Sample from Dirichlet distribution: Dir(α = [1, 1, ..., 1])
      - Produces diverse proportion combinations
      - Example: [0.42, 0.31, 0.18, 0.09] for 4 tissues
    
    Distribution:
      - 50% equiproportional
      - 50% Dirichlet random
    
  Augmentation Handling:
    - Mix samples from SAME augmentation version
    - Train across ALL 5 augmentation versions
    
  Training Data (80% mixtures + 20% pure):
    - Mixtures: Generate 4,000 per epoch (on-the-fly)
      - 50% with 3 tissues
      - 30% with 4 tissues
      - 20% with 5 tissues
    - Pure samples: 1,000 per epoch
    - Total per epoch: 5,000 training examples
    
  Validation Data (pre-generated):
    - 1,000 mixtures (fixed, saved to HDF5)
    - Balanced across 3/4/5 tissue scenarios
    
  Test Data (pre-generated):
    - 1,000 mixtures (fixed, saved to HDF5)
```

### Phase 3: Realistic Clinical Mixtures

**Objective**: Train on biologically realistic cfDNA compositions

**Configuration:**
```yaml
Phase 3 - Realistic Clinical Distributions (5+ tissues):
  Composition Templates:
    Template A (Healthy Baseline):
      - Blood cells: 75-85%
      - Liver: 5-10%
      - Lung: 2-5%
      - Intestine: 2-5%
      - Other tissues: <2% each
    
    Template B (Organ Damage):
      - Blood cells: 70-75%
      - Damaged organ (liver/lung/kidney): 10-20%
      - Other organs: 2-8% distributed
      - Trace tissues: <2%
    
    Template C (Multi-Organ Involvement):
      - Blood cells: 65-75%
      - Primary organ: 10-15%
      - Secondary organ: 5-10%
      - Tertiary organ: 3-7%
      - Trace tissues: <2%
    
  Tissue Selection:
    - Blood: ALWAYS included (dominant component)
    - Major organs: Randomly select 2-4 from [Liver, Lung, Kidney, Heart, Brain]
    - Minor tissues: Randomly select 2-3 from remaining 15 tissues
    - Total tissues per mixture: 6-9
    
  Proportion Generation:
    1. Sample blood proportion: Uniform(0.65, 0.85)
    2. Remaining (1 - blood_prop) distributed among other tissues
    3. Sample major organ proportions: Dirichlet with α=2 (slightly concentrated)
    4. Sample minor tissue proportions: Dirichlet with α=0.5 (sparse)
    5. Normalize all to sum = 1.0
    
  Augmentation Handling:
    - NOW mix across augmentation versions (simulate varying coverage in real cfDNA)
    - Each tissue component can come from different aug version
    - More realistic: blood might be high-coverage (aug0), organs lower (aug2-4)
    
  Training Data (80% mixtures + 20% pure):
    - Mixtures: Generate 8,000 per epoch
      - 40% Template A (healthy)
      - 40% Template B (organ damage)
      - 20% Template C (multi-organ)
    - Pure samples: 2,000 per epoch
    - Total per epoch: 10,000 training examples
    
  Validation Data (pre-generated):
    - 2,000 mixtures (fixed, saved to HDF5)
    - Balanced across templates A/B/C
    
  Test Data (pre-generated):
    - 2,000 mixtures (fixed, saved to HDF5)
```

### Mixture Generation Implementation

**Class Structure:**
```python
class MixtureGenerator:
    """
    Generate synthetic tissue mixtures from pure samples.
    
    Supports:
    - On-the-fly generation (training)
    - Pre-generated fixed sets (validation/test)
    - All three training phases
    """
    
    def __init__(self, hdf5_path, metadata_csv, split='train', phase=1, seed=42):
        """
        Args:
            hdf5_path: Path to methylation_dataset.h5
            metadata_csv: Path to combined_metadata.csv (with tissue labels)
            split: 'train', 'val', or 'test'
            phase: 1 (2-tissue), 2 (3-5 tissue), or 3 (realistic)
            seed: Random seed for reproducibility
        """
        self.hdf5_path = hdf5_path
        self.metadata = pd.read_csv(metadata_csv)
        self.split = split
        self.phase = phase
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load HDF5 file (keep handle open for fast access)
        self.h5_file = h5py.File(hdf5_path, 'r')
        
        # Index samples by tissue type and augmentation version
        self._build_tissue_index()
    
    def _build_tissue_index(self):
        """
        Create lookup tables:
        - tissue_to_samples: {tissue_id: [sample_indices]}
        - sample_to_tissue: {sample_idx: tissue_id}
        - aug_version: {sample_idx: aug_version}
        """
        self.tissue_to_samples = defaultdict(list)
        self.sample_to_tissue = {}
        self.aug_version = {}
        
        for idx, row in self.metadata.iterrows():
            tissue_id = row['tissue_label']  # Assuming column exists
            aug_ver = row['aug_version']
            
            self.tissue_to_samples[tissue_id].append(idx)
            self.sample_to_tissue[idx] = tissue_id
            self.aug_version[idx] = aug_ver
    
    def generate_mixture(self, n_tissues, proportions=None, aug_version=None):
        """
        Generate a single synthetic mixture.
        
        Args:
            n_tissues: Number of tissues to mix (2-9)
            proportions: Optional pre-specified proportions [n_tissues]
                        If None, will generate based on phase strategy
            aug_version: Optional augmentation version to use (0-4)
                        If None, will be randomly selected (Phase 1-2) or mixed (Phase 3)
        
        Returns:
            mixed_methylation: [51089] - mixed region means
            true_proportions: [22] - ground truth tissue proportions
            metadata: dict with mixing details
        """
        # Select different tissues (CRITICAL: ensure they differ!)
        selected_tissues = self._select_different_tissues(n_tissues)
        
        # Select samples from each tissue
        if aug_version is not None:
            # Use same aug version for all (Phase 1-2)
            selected_samples = [self._select_sample_from_tissue(t, aug_version) 
                               for t in selected_tissues]
        else:
            # Mixed aug versions (Phase 3)
            selected_samples = [self._select_sample_from_tissue(t) 
                               for t in selected_tissues]
        
        # Generate or use provided proportions
        if proportions is None:
            proportions = self._generate_proportions(n_tissues)
        
        # Load region means for each sample
        region_means = []
        for sample_idx in selected_samples:
            methylation = self.h5_file['methylation'][sample_idx]  # [51089, 150]
            
            # Compute region means (same as model does)
            valid_mask = (methylation != 2).astype(float)
            region_mean = np.sum(methylation * valid_mask, axis=1) / (np.sum(valid_mask, axis=1) + 1e-8)
            region_means.append(region_mean)
        
        region_means = np.array(region_means)  # [n_tissues, 51089]
        
        # Linear mixing
        mixed_methylation = np.sum(region_means * proportions[:, np.newaxis], axis=0)
        # Shape: [51089]
        
        # Convert to 22-dimensional proportion vector
        true_proportions = np.zeros(22, dtype=np.float32)
        for tissue_id, prop in zip(selected_tissues, proportions):
            true_proportions[tissue_id] = prop
        
        # Metadata for logging
        metadata = {
            'n_tissues': n_tissues,
            'tissue_ids': selected_tissues,
            'sample_indices': selected_samples,
            'proportions': proportions.tolist(),
            'aug_versions': [self.aug_version[s] for s in selected_samples]
        }
        
        return mixed_methylation, true_proportions, metadata
    
    def _select_different_tissues(self, n_tissues):
        """
        CRITICAL: Select N DIFFERENT tissue types.
        
        Returns:
            List of n_tissues unique tissue IDs (0-21)
        """
        all_tissues = list(range(22))
        selected = random.sample(all_tissues, n_tissues)
        return selected
    
    def _select_sample_from_tissue(self, tissue_id, aug_version=None):
        """
        Select a random sample from specified tissue and augmentation version.
        
        Args:
            tissue_id: Tissue type (0-21)
            aug_version: Optional aug version (0-4), if None select randomly
        
        Returns:
            sample_idx: Index into HDF5 dataset
        """
        available_samples = self.tissue_to_samples[tissue_id]
        
        if aug_version is not None:
            # Filter by augmentation version
            available_samples = [s for s in available_samples 
                                if self.aug_version[s] == aug_version]
        
        if len(available_samples) == 0:
            raise ValueError(f"No samples available for tissue {tissue_id}, aug {aug_version}")
        
        return random.choice(available_samples)
    
    def _generate_proportions(self, n_tissues):
        """
        Generate proportions based on phase strategy.
        
        Returns:
            proportions: [n_tissues] array summing to 1.0
        """
        if self.phase == 1:
            # Phase 1: Random selection among predefined strategies
            strategies = [
                [0.5, 0.5],  # 50-50
                [0.6, 0.4],  # 60-40
                [0.4, 0.6],  # 40-60
                [0.7, 0.3],  # 70-30
                [0.3, 0.7],  # 30-70
                [0.8, 0.2],  # 80-20
                [0.2, 0.8],  # 20-80
            ]
            return np.array(random.choice(strategies), dtype=np.float32)
        
        elif self.phase == 2:
            # Phase 2: 50% equiproportional, 50% Dirichlet
            if random.random() < 0.5:
                # Equiproportional
                proportions = np.ones(n_tissues, dtype=np.float32) / n_tissues
            else:
                # Dirichlet with α=1 (uniform)
                proportions = np.random.dirichlet(np.ones(n_tissues)).astype(np.float32)
            return proportions
        
        elif self.phase == 3:
            # Phase 3: Realistic distributions
            # Note: This is simplified - actual implementation will use templates
            
            # Blood proportion (always first tissue)
            blood_prop = np.random.uniform(0.65, 0.85)
            remaining_prop = 1.0 - blood_prop
            
            # Distribute remaining among other tissues
            if n_tissues == 1:
                proportions = np.array([blood_prop], dtype=np.float32)
            else:
                # Major organs (higher α)
                n_major = min(3, n_tissues - 1)
                major_props = np.random.dirichlet(np.ones(n_major) * 2) * remaining_prop * 0.7
                
                # Minor tissues (lower α)
                n_minor = n_tissues - 1 - n_major
                if n_minor > 0:
                    minor_props = np.random.dirichlet(np.ones(n_minor) * 0.5) * remaining_prop * 0.3
                    other_props = np.concatenate([major_props, minor_props])
                else:
                    other_props = major_props
                
                proportions = np.concatenate([[blood_prop], other_props]).astype(np.float32)
            
            # Ensure sum to 1
            proportions = proportions / proportions.sum()
            return proportions
    
    def generate_batch(self, batch_size, pure_sample_ratio=0.2):
        """
        Generate a batch of mixtures (for training).
        
        Args:
            batch_size: Number of samples in batch
            pure_sample_ratio: Fraction of batch that should be pure samples
        
        Returns:
            mixed_methylation: [batch_size, 51089]
            true_proportions: [batch_size, 22]
        """
        n_mixtures = int(batch_size * (1 - pure_sample_ratio))
        n_pure = batch_size - n_mixtures
        
        methylation_batch = []
        proportions_batch = []
        
        # Generate mixtures
        for _ in range(n_mixtures):
            if self.phase == 1:
                n_tissues = 2
            elif self.phase == 2:
                n_tissues = random.choice([3, 4, 5])
            else:  # phase 3
                n_tissues = random.randint(6, 9)
            
            mixed_meth, true_prop, _ = self.generate_mixture(n_tissues)
            methylation_batch.append(mixed_meth)
            proportions_batch.append(true_prop)
        
        # Add pure samples (100% one tissue)
        for _ in range(n_pure):
            tissue_id = random.randint(0, 21)
            sample_idx = self._select_sample_from_tissue(tissue_id)
            
            # Load region means
            methylation = self.h5_file['methylation'][sample_idx]
            valid_mask = (methylation != 2).astype(float)
            region_mean = np.sum(methylation * valid_mask, axis=1) / (np.sum(valid_mask, axis=1) + 1e-8)
            
            # One-hot proportion
            true_prop = np.zeros(22, dtype=np.float32)
            true_prop[tissue_id] = 1.0
            
            methylation_batch.append(region_mean)
            proportions_batch.append(true_prop)
        
        # Convert to tensors
        methylation_batch = torch.tensor(np.array(methylation_batch), dtype=torch.float32)
        proportions_batch = torch.tensor(np.array(proportions_batch), dtype=torch.float32)
        
        return methylation_batch, proportions_batch
```

**Pre-generate Validation/Test Sets:**
```python
def create_fixed_mixture_sets(hdf5_path, metadata_csv, phase, output_dir, seed_val=42, seed_test=43):
    """
    Pre-generate fixed validation and test mixture sets.
    
    Args:
        hdf5_path: Path to methylation_dataset.h5
        metadata_csv: Path to combined_metadata.csv
        phase: 1, 2, or 3
        output_dir: Where to save HDF5 files
        seed_val: Seed for validation set
        seed_test: Seed for test set
    
    Creates:
        {output_dir}/phase{phase}_validation_mixtures.h5
        {output_dir}/phase{phase}_test_mixtures.h5
    """
    # Determine number of mixtures per set
    if phase == 1:
        n_mixtures = 500
    elif phase == 2:
        n_mixtures = 1000
    else:
        n_mixtures = 2000
    
    # Generate validation set
    print(f"Generating {n_mixtures} validation mixtures for Phase {phase}...")
    val_generator = MixtureGenerator(hdf5_path, metadata_csv, split='val', phase=phase, seed=seed_val)
    val_mixtures, val_proportions, val_metadata = [], [], []
    
    for i in range(n_mixtures):
        if phase == 1:
            n_tissues = 2
        elif phase == 2:
            n_tissues = random.choice([3, 4, 5])
        else:
            n_tissues = random.randint(6, 9)
        
        mixed_meth, true_prop, meta = val_generator.generate_mixture(n_tissues)
        val_mixtures.append(mixed_meth)
        val_proportions.append(true_prop)
        val_metadata.append(meta)
    
    # Save to HDF5
    val_file = os.path.join(output_dir, f'phase{phase}_validation_mixtures.h5')
    with h5py.File(val_file, 'w') as f:
        f.create_dataset('mixed_methylation', data=np.array(val_mixtures), dtype=np.float32)
        f.create_dataset('true_proportions', data=np.array(val_proportions), dtype=np.float32)
        f.create_dataset('metadata', data=json.dumps(val_metadata))
    
    print(f"✓ Saved validation set to {val_file}")
    
    # Generate test set (same process with different seed)
    print(f"Generating {n_mixtures} test mixtures for Phase {phase}...")
    test_generator = MixtureGenerator(hdf5_path, metadata_csv, split='test', phase=phase, seed=seed_test)
    test_mixtures, test_proportions, test_metadata = [], [], []
    
    for i in range(n_mixtures):
        if phase == 1:
            n_tissues = 2
        elif phase == 2:
            n_tissues = random.choice([3, 4, 5])
        else:
            n_tissues = random.randint(6, 9)
        
        mixed_meth, true_prop, meta = test_generator.generate_mixture(n_tissues)
        test_mixtures.append(mixed_meth)
        test_proportions.append(true_prop)
        test_metadata.append(meta)
    
    # Save to HDF5
    test_file = os.path.join(output_dir, f'phase{phase}_test_mixtures.h5')
    with h5py.File(test_file, 'w') as f:
        f.create_dataset('mixed_methylation', data=np.array(test_mixtures), dtype=np.float32)
        f.create_dataset('true_proportions', data=np.array(test_proportions), dtype=np.float32)
        f.create_dataset('metadata', data=json.dumps(test_metadata))
    
    print(f"✓ Saved test set to {test_file}")
```

---

## Training Pipeline Design

### Progressive Training Schedule

**Strategy**: Train sequentially through phases, each building on previous phase

```
Phase 1 (2-Tissue) → Phase 2 (3-5 Tissue) → Phase 3 (Realistic)
     ↓                      ↓                        ↓
Checkpoint 1          Checkpoint 2             Checkpoint 3 (Final)
```

### Training Configuration

**Phase 1 Configuration:**
```yaml
# config_phase1_2tissue.yaml

data:
  hdf5_path: '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/methylation_dataset.h5'
  metadata_csv: '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/combined_metadata.csv'
  validation_h5: '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/phase1_validation_mixtures.h5'
  test_h5: '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/phase1_test_mixtures.h5'

model:
  n_regions: 51089
  hidden_size: 512
  num_classes: 22
  dropout: 0.1
  pretrained_checkpoint: '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/fullgenome_results/checkpoints/checkpoint_best_acc.pt'

training:
  num_epochs: 30
  batch_size: 4
  gradient_accumulation_steps: 8  # Effective batch = 32
  learning_rate: 1.0e-5  # Same as original training
  weight_decay: 0.01
  warmup_ratio: 0.1
  num_workers: 12
  validation_frequency: 5  # Validate every 5 epochs
  
  # Mixture generation
  phase: 1  # 2-tissue mixtures
  pure_sample_ratio: 0.2  # 20% pure samples
  mixtures_per_epoch: 2500  # Total training examples per epoch

optimizer:
  type: 'AdamW'
  betas: [0.9, 0.999]
  eps: 1.0e-8

scheduler:
  type: 'cosine'
  warmup_steps: 1000

loss:
  type: 'mse'  # MSE on proportions

random_seed: 42

output:
  save_dir: '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase1'
  log_every_n_steps: 100
  save_best_model: True
  save_last_model: True
```

**Phase 2 Configuration:**
```yaml
# config_phase2_multitissue.yaml
# (Same structure, different values)

model:
  pretrained_checkpoint: '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase1/checkpoints/checkpoint_best.pt'

training:
  phase: 2  # 3-5 tissue mixtures
  mixtures_per_epoch: 5000

output:
  save_dir: '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase2'
```

**Phase 3 Configuration:**
```yaml
# config_phase3_realistic.yaml

model:
  pretrained_checkpoint: '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase2/checkpoints/checkpoint_best.pt'

training:
  phase: 3  # Realistic clinical distributions
  mixtures_per_epoch: 10000

output:
  save_dir: '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase3'
```

### Training Loop Implementation

```python
def train_mixture_deconvolution(config):
    """
    Main training loop for mixture deconvolution.
    
    Args:
        config: Configuration dictionary (from YAML)
    """
    # Set random seeds
    set_random_seeds(config['random_seed'])
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if config['model']['pretrained_checkpoint']:
        model = load_pretrained_model(config['model']['pretrained_checkpoint'], device)
        print("✓ Loaded pre-trained model")
    else:
        model = TissueBERTDeconvolution(**config['model']).to(device)
        print("✓ Initialized model from scratch")
    
    # Initialize mixture generator
    train_generator = MixtureGenerator(
        hdf5_path=config['data']['hdf5_path'],
        metadata_csv=config['data']['metadata_csv'],
        split='train',
        phase=config['training']['phase'],
        seed=config['random_seed']
    )
    
    # Load validation data (pre-generated)
    val_data = load_validation_mixtures(config['data']['validation_h5'])
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=config['optimizer']['betas'],
        eps=config['optimizer']['eps']
    )
    
    # Learning rate scheduler
    total_steps = (config['training']['mixtures_per_epoch'] // 
                   (config['training']['batch_size'] * config['training']['gradient_accumulation_steps'])) * \
                   config['training']['num_epochs']
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config['training']['warmup_ratio']),
        num_training_steps=total_steps
    )
    
    # Training metrics
    best_val_mae = float('inf')
    train_losses = []
    val_maes = []
    
    # Training loop
    for epoch in range(config['training']['num_epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        print(f"{'='*60}")
        
        # Train for one epoch
        model.train()
        epoch_loss = 0
        n_batches = config['training']['mixtures_per_epoch'] // config['training']['batch_size']
        
        for batch_idx in range(n_batches):
            # Generate batch (on-the-fly)
            mixed_meth, true_prop = train_generator.generate_batch(
                batch_size=config['training']['batch_size'],
                pure_sample_ratio=config['training']['pure_sample_ratio']
            )
            
            mixed_meth = mixed_meth.to(device)
            true_prop = true_prop.to(device)
            
            # Forward pass (need to expand to match model input format)
            # Model expects [batch, n_regions, 150], but we have [batch, n_regions]
            # We'll modify this in actual implementation
            
            # Placeholder: expand region means to full format
            # In practice, we'd store full methylation patterns for mixing
            
            predicted_prop = model(mixed_meth.unsqueeze(-1).expand(-1, -1, 150))
            
            # Compute loss
            loss = mixture_mse_loss(predicted_prop, true_prop)
            loss = loss / config['training']['gradient_accumulation_steps']
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % config['training']['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            
            # Logging
            if batch_idx % config['output']['log_every_n_steps'] == 0:
                print(f"  Batch {batch_idx}/{n_batches}, Loss: {loss.item():.6f}, "
                      f"LR: {scheduler.get_last_lr()[0]:.6e}")
        
        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)
        print(f"\nEpoch {epoch+1} - Avg Train Loss: {avg_train_loss:.6f}")
        
        # Validation
        if (epoch + 1) % config['training']['validation_frequency'] == 0:
            print(f"\n{'='*60}")
            print(f"Validation")
            print(f"{'='*60}")
            
            val_mae, val_metrics = validate_model(model, val_data, device)
            val_maes.append(val_mae)
            
            print(f"Validation MAE: {val_mae:.4f}")
            print(f"Per-tissue MAE: {val_metrics['per_tissue_mae']}")
            
            # Save best model
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                save_checkpoint(
                    model, optimizer, epoch, val_mae,
                    os.path.join(config['output']['save_dir'], 'checkpoints', 'checkpoint_best.pt')
                )
                print(f"✓ Saved best model (MAE: {val_mae:.4f})")
        
        # Save last checkpoint
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, avg_train_loss,
                os.path.join(config['output']['save_dir'], 'checkpoints', f'checkpoint_epoch{epoch+1}.pt')
            )
    
    # Final save
    save_checkpoint(
        model, optimizer, epoch, avg_train_loss,
        os.path.join(config['output']['save_dir'], 'checkpoints', 'checkpoint_last.pt')
    )
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Validation MAE: {best_val_mae:.4f}")
    print(f"{'='*60}")
```

### Validation Function

```python
def validate_model(model, val_data, device):
    """
    Validate model on pre-generated validation set.
    
    Args:
        model: TissueBERTDeconvolution model
        val_data: Dictionary with 'mixed_methylation' and 'true_proportions'
        device: 'cuda' or 'cpu'
    
    Returns:
        overall_mae: Mean Absolute Error across all predictions
        metrics: Dictionary with detailed metrics
    """
    model.eval()
    
    mixed_meth = torch.tensor(val_data['mixed_methylation'], dtype=torch.float32).to(device)
    true_prop = torch.tensor(val_data['true_proportions'], dtype=torch.float32).to(device)
    
    # Batch processing for memory efficiency
    batch_size = 32
    n_samples = mixed_meth.shape[0]
    predictions = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_meth = mixed_meth[i:i+batch_size]
            
            # Forward pass
            batch_pred = model(batch_meth.unsqueeze(-1).expand(-1, -1, 150))
            predictions.append(batch_pred.cpu())
    
    predictions = torch.cat(predictions, dim=0)
    true_prop = true_prop.cpu()
    
    # Compute metrics
    mae = torch.abs(predictions - true_prop).mean().item()
    
    # Per-tissue MAE
    per_tissue_mae = torch.abs(predictions - true_prop).mean(dim=0).numpy()
    
    # Correlation per tissue
    correlations = []
    for tissue_idx in range(22):
        corr = np.corrcoef(predictions[:, tissue_idx].numpy(), 
                          true_prop[:, tissue_idx].numpy())[0, 1]
        correlations.append(corr)
    
    # Detection threshold analysis (for minor components)
    # Can model detect tissues with <5% proportion?
    minor_mask = true_prop < 0.05
    major_mask = true_prop >= 0.05
    
    minor_mae = torch.abs(predictions[minor_mask] - true_prop[minor_mask]).mean().item()
    major_mae = torch.abs(predictions[major_mask] - true_prop[major_mask]).mean().item()
    
    metrics = {
        'overall_mae': mae,
        'per_tissue_mae': per_tissue_mae.tolist(),
        'correlations': correlations,
        'minor_component_mae': minor_mae,  # <5%
        'major_component_mae': major_mae,  # >=5%
    }
    
    return mae, metrics
```

---

## Evaluation Framework

### Metrics to Track

**Primary Metrics:**
1. **Mean Absolute Error (MAE)**: Average |predicted - true| across all tissues
2. **Per-Tissue MAE**: MAE for each of 22 tissue types
3. **Correlation**: Pearson correlation between predicted and true proportions per tissue
4. **Detection Threshold**: Can model detect <5%, <10%, <20% components?

**Secondary Metrics:**
5. **Top-K Accuracy**: Is true dominant tissue in top-K predictions?
6. **RMSE**: Root Mean Squared Error (penalizes large errors more)
7. **R² Score**: Proportion of variance explained

### Evaluation Implementation

```python
def comprehensive_evaluation(model, test_data, output_dir, phase):
    """
    Comprehensive evaluation on test set with visualizations.
    
    Args:
        model: Trained TissueBERTDeconvolution model
        test_data: Test mixture dataset
        output_dir: Where to save results
        phase: 1, 2, or 3
    
    Generates:
        - Scatter plots (predicted vs true) per tissue
        - Confusion matrix for dominant tissue
        - Error distribution histograms
        - Summary report
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Load test data
    mixed_meth = torch.tensor(test_data['mixed_methylation'], dtype=torch.float32).to(device)
    true_prop = torch.tensor(test_data['true_proportions'], dtype=torch.float32).to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions = []
        batch_size = 32
        for i in range(0, len(mixed_meth), batch_size):
            batch = mixed_meth[i:i+batch_size]
            pred = model(batch.unsqueeze(-1).expand(-1, -1, 150))
            predictions.append(pred.cpu())
    
    predictions = torch.cat(predictions, dim=0).numpy()
    true_prop = true_prop.cpu().numpy()
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'metrics'), exist_ok=True)
    
    # 1. Overall MAE
    overall_mae = np.abs(predictions - true_prop).mean()
    
    # 2. Per-tissue metrics
    per_tissue_mae = np.abs(predictions - true_prop).mean(axis=0)
    per_tissue_corr = [np.corrcoef(predictions[:, i], true_prop[:, i])[0, 1] 
                       for i in range(22)]
    
    # 3. Detection thresholds
    thresholds = [0.05, 0.10, 0.20]
    detection_metrics = {}
    
    for thresh in thresholds:
        mask = true_prop >= thresh
        if mask.sum() > 0:
            detection_mae = np.abs(predictions[mask] - true_prop[mask]).mean()
            detection_metrics[f'mae_above_{int(thresh*100)}pct'] = detection_mae
    
    # 4. Top-K accuracy (dominant tissue)
    dominant_true = np.argmax(true_prop, axis=1)
    top_k_accuracy = {}
    
    for k in [1, 3, 5]:
        top_k_pred = np.argsort(predictions, axis=1)[:, -k:]
        top_k_acc = np.mean([dominant_true[i] in top_k_pred[i] 
                            for i in range(len(dominant_true))])
        top_k_accuracy[f'top_{k}'] = top_k_acc
    
    # 5. RMSE and R²
    rmse = np.sqrt(np.mean((predictions - true_prop) ** 2))
    
    # R² per tissue
    r2_scores = []
    for i in range(22):
        ss_res = np.sum((true_prop[:, i] - predictions[:, i]) ** 2)
        ss_tot = np.sum((true_prop[:, i] - true_prop[:, i].mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r2_scores.append(r2)
    
    # Compile all metrics
    metrics = {
        'phase': phase,
        'overall_mae': float(overall_mae),
        'overall_rmse': float(rmse),
        'per_tissue_mae': per_tissue_mae.tolist(),
        'per_tissue_correlation': per_tissue_corr,
        'per_tissue_r2': r2_scores,
        'detection_thresholds': detection_metrics,
        'top_k_accuracy': top_k_accuracy,
    }
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics', f'phase{phase}_test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate visualizations
    plot_scatter_per_tissue(predictions, true_prop, output_dir, phase)
    plot_error_distribution(predictions, true_prop, output_dir, phase)
    plot_confusion_matrix_dominant(predictions, true_prop, output_dir, phase)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Phase {phase} Test Evaluation Summary")
    print(f"{'='*60}")
    print(f"Overall MAE: {overall_mae:.4f}")
    print(f"Overall RMSE: {rmse:.4f}")
    print(f"Top-1 Accuracy (dominant tissue): {top_k_accuracy['top_1']:.2%}")
    print(f"Top-3 Accuracy: {top_k_accuracy['top_3']:.2%}")
    print(f"\nDetection Thresholds:")
    for key, val in detection_metrics.items():
        print(f"  {key}: {val:.4f}")
    print(f"\nTop 5 Best Tissues (by MAE):")
    best_tissues = np.argsort(per_tissue_mae)[:5]
    for tissue_idx in best_tissues:
        print(f"  Tissue {tissue_idx}: MAE={per_tissue_mae[tissue_idx]:.4f}, "
              f"Corr={per_tissue_corr[tissue_idx]:.3f}")
    print(f"\nTop 5 Worst Tissues (by MAE):")
    worst_tissues = np.argsort(per_tissue_mae)[-5:]
    for tissue_idx in worst_tissues:
        print(f"  Tissue {tissue_idx}: MAE={per_tissue_mae[tissue_idx]:.4f}, "
              f"Corr={per_tissue_corr[tissue_idx]:.3f}")
    print(f"{'='*60}\n")
    
    return metrics
```

### Visualization Functions

```python
def plot_scatter_per_tissue(predictions, true_prop, output_dir, phase):
    """
    Create scatter plots of predicted vs true proportions for each tissue.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(4, 6, figsize=(24, 16))
    axes = axes.flatten()
    
    for tissue_idx in range(22):
        ax = axes[tissue_idx]
        
        ax.scatter(true_prop[:, tissue_idx], predictions[:, tissue_idx], 
                  alpha=0.5, s=10)
        ax.plot([0, 1], [0, 1], 'r--', lw=2)
        
        # Compute correlation
        corr = np.corrcoef(predictions[:, tissue_idx], true_prop[:, tissue_idx])[0, 1]
        mae = np.abs(predictions[:, tissue_idx] - true_prop[:, tissue_idx]).mean()
        
        ax.set_xlabel('True Proportion')
        ax.set_ylabel('Predicted Proportion')
        ax.set_title(f'Tissue {tissue_idx}\nCorr={corr:.3f}, MAE={mae:.3f}')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(22, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', f'phase{phase}_scatter_per_tissue.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_distribution(predictions, true_prop, output_dir, phase):
    """
    Plot distribution of prediction errors.
    """
    import matplotlib.pyplot as plt
    
    errors = predictions - true_prop
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall error distribution
    axes[0].hist(errors.flatten(), bins=100, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Prediction Error (Predicted - True)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Phase {phase}: Overall Error Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Absolute error by true proportion
    abs_errors = np.abs(errors)
    axes[1].scatter(true_prop.flatten(), abs_errors.flatten(), alpha=0.1, s=5)
    axes[1].set_xlabel('True Proportion')
    axes[1].set_ylabel('Absolute Error')
    axes[1].set_title(f'Phase {phase}: Error vs True Proportion')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', f'phase{phase}_error_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix_dominant(predictions, true_prop, output_dir, phase):
    """
    Confusion matrix for dominant tissue prediction.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Get dominant tissues
    dominant_true = np.argmax(true_prop, axis=1)
    dominant_pred = np.argmax(predictions, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(dominant_true, dominant_pred, labels=range(22))
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues', 
                xticklabels=range(22), yticklabels=range(22), ax=ax)
    ax.set_xlabel('Predicted Dominant Tissue')
    ax.set_ylabel('True Dominant Tissue')
    ax.set_title(f'Phase {phase}: Dominant Tissue Confusion Matrix (Normalized)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figures', f'phase{phase}_confusion_matrix.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
```

---

## Implementation Timeline

### Week-by-Week Breakdown

**Week 1: Setup and Phase 1 Preparation**
```
Days 1-2: Environment Setup
├── Create directory structure
├── Implement MixtureGenerator class
├── Test on small subset (10 samples)
└── Verify linear mixing makes sense

Days 3-4: Pre-generate Validation/Test Sets
├── Generate Phase 1 validation set (500 mixtures)
├── Generate Phase 1 test set (500 mixtures)
├── Save to HDF5
└── Verify data integrity

Days 5-7: Model Modification
├── Implement TissueBERTDeconvolution class
├── Load pre-trained checkpoint
├── Test forward pass with mixed samples
├── Implement training loop for Phase 1
└── Test on small batch (dry run)

Deliverable: Phase 1 training pipeline ready
```

**Week 2: Phase 1 Training**
```
Days 1-3: Initial Training
├── Launch Phase 1 training (30 epochs)
├── Monitor training metrics
├── Debug any issues
└── Evaluate on validation set every 5 epochs

Days 4-5: Analysis and Tuning
├── Analyze Phase 1 results
├── Identify problematic tissue pairs
├── Tune hyperparameters if needed
└── Re-train if necessary

Days 6-7: Phase 1 Evaluation
├── Run comprehensive evaluation on test set
├── Generate all plots and metrics
├── Document results
└── Prepare Phase 2 setup

Deliverable: Phase 1 trained model (MAE <5%)
```

**Week 3: Phase 2 Training**
```
Days 1-2: Phase 2 Preparation
├── Pre-generate Phase 2 validation/test sets (1000 each)
├── Update configuration for 3-5 tissue mixtures
└── Test mixture generator for Phase 2

Days 3-5: Phase 2 Training
├── Launch Phase 2 training (load Phase 1 checkpoint)
├── Monitor training (30 epochs)
└── Validate every 5 epochs

Days 6-7: Phase 2 Evaluation
├── Comprehensive test evaluation
├── Compare with Phase 1 performance
├── Analyze error patterns
└── Prepare Phase 3 setup

Deliverable: Phase 2 trained model
```

**Week 4: Phase 3 Training and Final Evaluation**
```
Days 1-2: Phase 3 Preparation
├── Pre-generate Phase 3 validation/test sets (2000 each)
├── Implement realistic distribution templates
└── Test cross-augmentation mixing

Days 3-5: Phase 3 Training
├── Launch Phase 3 training (load Phase 2 checkpoint)
├── Monitor training (30 epochs)
└── Validate every 5 epochs

Days 6-7: Final Evaluation and Documentation
├── Comprehensive test evaluation across all phases
├── Generate comparison figures
├── Write final evaluation report
├── Prepare model for PDAC application
└── Document complete pipeline

Deliverable: Production-ready mixture deconvolution model
```

### Estimated Compute Time

**Training Time per Epoch:**
- Phase 1 (2,500 examples): ~60 seconds
- Phase 2 (5,000 examples): ~120 seconds
- Phase 3 (10,000 examples): ~240 seconds

**Total Training Time:**
- Phase 1 (30 epochs): ~30 minutes
- Phase 2 (30 epochs): ~60 minutes
- Phase 3 (30 epochs): ~120 minutes
- **Total: ~3.5 hours**

**Hardware: NVIDIA A100 40GB**

---

## File Structure

```
/home/chattopa/data_storage/MethAtlas_WGBSanalysis/
├── training_dataset/
│   ├── methylation_dataset.h5 (existing - 5.8GB)
│   ├── combined_metadata.csv (existing)
│   │
│   ├── mixture_datasets/  (NEW)
│   │   ├── phase1_validation_mixtures.h5
│   │   ├── phase1_test_mixtures.h5
│   │   ├── phase2_validation_mixtures.h5
│   │   ├── phase2_test_mixtures.h5
│   │   ├── phase3_validation_mixtures.h5
│   │   └── phase3_test_mixtures.h5
│   │
│   └── splits/
│       ├── train_files.csv (existing)
│       ├── val_files.csv (existing)
│       └── test_files.csv (existing)
│
├── fullgenome_results/  (existing single-tissue model)
│   └── checkpoints/
│       └── checkpoint_best_acc.pt (97.8% accuracy - starting point)
│
├── mixture_deconvolution_results/  (NEW)
│   ├── phase1_2tissue/
│   │   ├── checkpoints/
│   │   │   ├── checkpoint_best.pt
│   │   │   ├── checkpoint_last.pt
│   │   │   └── checkpoint_epoch{5,10,15,...}.pt
│   │   ├── logs/
│   │   │   ├── training_log.csv
│   │   │   ├── events.out.tfevents.* (TensorBoard)
│   │   │   └── slurm_*.out
│   │   ├── evaluation/
│   │   │   ├── metrics/
│   │   │   │   └── phase1_test_metrics.json
│   │   │   └── figures/
│   │   │       ├── phase1_scatter_per_tissue.png
│   │   │       ├── phase1_error_distribution.png
│   │   │       └── phase1_confusion_matrix.png
│   │   └── config_phase1.yaml
│   │
│   ├── phase2_multitissue/
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   ├── evaluation/
│   │   └── config_phase2.yaml
│   │
│   ├── phase3_realistic/
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   ├── evaluation/
│   │   └── config_phase3.yaml
│   │
│   └── final_model/
│       ├── model_weights.pt (final Phase 3 checkpoint)
│       ├── model_architecture.py
│       ├── inference_script.py
│       └── README.md
│
└── scripts/  (NEW)
    ├── mixture_generation/
    │   ├── generate_mixtures.py
    │   ├── mixture_generator.py
    │   └── validate_mixing.py
    │
    ├── training/
    │   ├── train_phase1.py
    │   ├── train_phase2.py
    │   ├── train_phase3.py
    │   ├── model_deconvolution.py
    │   ├── dataloader_mixtures.py
    │   └── utils_training.py
    │
    ├── evaluation/
    │   ├── evaluate_model.py
    │   ├── plot_results.py
    │   └── compute_metrics.py
    │
    └── configs/
        ├── config_phase1_2tissue.yaml
        ├── config_phase2_multitissue.yaml
        └── config_phase3_realistic.yaml
```

---

## Detailed Step-by-Step Execution

### Step 1: Environment Setup (Day 1)

**1.1 Create Directory Structure**
```bash
cd /home/chattopa/data_storage/MethAtlas_WGBSanalysis

# Create new directories
mkdir -p mixture_deconvolution_results/{phase1_2tissue,phase2_multitissue,phase3_realistic}/checkpoints
mkdir -p mixture_deconvolution_results/{phase1_2tissue,phase2_multitissue,phase3_realistic}/logs
mkdir -p mixture_deconvolution_results/{phase1_2tissue,phase2_multitissue,phase3_realistic}/evaluation/{metrics,figures}
mkdir -p mixture_deconvolution_results/final_model

mkdir -p training_dataset/mixture_datasets

mkdir -p scripts/{mixture_generation,training,evaluation,configs}
```

**1.2 Verify Data Availability**
```bash
# Check existing data
ls -lh training_dataset/methylation_dataset.h5
ls -lh combined_metadata.csv
ls -lh fullgenome_results/checkpoints/checkpoint_best_acc.pt

# Verify HDF5 structure
python3 << EOF
import h5py
with h5py.File('training_dataset/methylation_dataset.h5', 'r') as f:
    print("Dataset keys:", list(f.keys()))
    print("Methylation shape:", f['methylation'].shape)
    print("Tissue labels shape:", f['tissue_labels'].shape)
EOF
```

**1.3 Create Requirements File**
```bash
cat > scripts/requirements.txt << EOF
torch>=2.0.0
numpy>=1.24.0
h5py>=3.8.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
pyyaml>=6.0
tqdm>=4.65.0
tensorboard>=2.13.0
EOF

# Install (if needed)
pip install -r scripts/requirements.txt --break-system-packages
```

---

### Step 2: Implement Mixture Generator (Days 1-2)

**2.1 Create mixture_generator.py**

Save the MixtureGenerator class implementation (provided in Section 3) to:
```
scripts/mixture_generation/mixture_generator.py
```

**2.2 Test Mixture Generation**
```python
# scripts/mixture_generation/test_mixing.py

import sys
sys.path.append('/home/chattopa/data_storage/MethAtlas_WGBSanalysis/scripts/mixture_generation')

from mixture_generator import MixtureGenerator
import numpy as np

# Initialize generator
generator = MixtureGenerator(
    hdf5_path='/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/methylation_dataset.h5',
    metadata_csv='/home/chattopa/data_storage/MethAtlas_WGBSanalysis/combined_metadata.csv',
    split='train',
    phase=1,
    seed=42
)

# Test single mixture
print("Testing 2-tissue mixture generation...")
mixed_meth, true_prop, metadata = generator.generate_mixture(n_tissues=2)

print(f"Mixed methylation shape: {mixed_meth.shape}")
print(f"True proportions shape: {true_prop.shape}")
print(f"Proportions sum: {true_prop.sum():.6f}")
print(f"Non-zero tissues: {np.sum(true_prop > 0)}")
print(f"Metadata: {metadata}")

# Test batch generation
print("\nTesting batch generation...")
batch_meth, batch_prop = generator.generate_batch(batch_size=8, pure_sample_ratio=0.2)
print(f"Batch methylation shape: {batch_meth.shape}")
print(f"Batch proportions shape: {batch_prop.shape}")
print(f"All proportions sum to 1: {np.allclose(batch_prop.sum(axis=1), 1.0)}")

print("\n✓ Mixture generation test passed!")
```

Run test:
```bash
cd /home/chattopa/data_storage/MethAtlas_WGBSanalysis
python3 scripts/mixture_generation/test_mixing.py
```

---

### Step 3: Pre-generate Validation/Test Sets (Days 3-4)

**3.1 Create Generation Script**
```python
# scripts/mixture_generation/generate_mixtures.py

import os
import sys
import h5py
import numpy as np
import json
import random
from tqdm import tqdm

sys.path.append('/home/chattopa/data_storage/MethAtlas_WGBSanalysis/scripts/mixture_generation')
from mixture_generator import MixtureGenerator

def generate_fixed_sets(phase, n_val, n_test, output_dir):
    """
    Generate fixed validation and test sets for a given phase.
    """
    hdf5_path = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/methylation_dataset.h5'
    metadata_csv = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/combined_metadata.csv'
    
    # Validation set (seed=42)
    print(f"Generating Phase {phase} validation set ({n_val} mixtures)...")
    val_generator = MixtureGenerator(hdf5_path, metadata_csv, split='val', phase=phase, seed=42)
    
    val_mixtures = []
    val_proportions = []
    val_metadata = []
    
    for i in tqdm(range(n_val)):
        if phase == 1:
            n_tissues = 2
        elif phase == 2:
            n_tissues = random.choice([3, 4, 5])
        else:  # phase 3
            n_tissues = random.randint(6, 9)
        
        mixed_meth, true_prop, meta = val_generator.generate_mixture(n_tissues)
        val_mixtures.append(mixed_meth)
        val_proportions.append(true_prop)
        val_metadata.append(meta)
    
    val_file = os.path.join(output_dir, f'phase{phase}_validation_mixtures.h5')
    with h5py.File(val_file, 'w') as f:
        f.create_dataset('mixed_methylation', data=np.array(val_mixtures), dtype=np.float32, compression='gzip')
        f.create_dataset('true_proportions', data=np.array(val_proportions), dtype=np.float32, compression='gzip')
        f.attrs['metadata'] = json.dumps(val_metadata)
        f.attrs['phase'] = phase
        f.attrs['n_samples'] = n_val
        f.attrs['seed'] = 42
    
    print(f"✓ Saved validation set to {val_file}")
    
    # Test set (seed=43)
    print(f"\nGenerating Phase {phase} test set ({n_test} mixtures)...")
    test_generator = MixtureGenerator(hdf5_path, metadata_csv, split='test', phase=phase, seed=43)
    
    test_mixtures = []
    test_proportions = []
    test_metadata = []
    
    for i in tqdm(range(n_test)):
        if phase == 1:
            n_tissues = 2
        elif phase == 2:
            n_tissues = random.choice([3, 4, 5])
        else:  # phase 3
            n_tissues = random.randint(6, 9)
        
        mixed_meth, true_prop, meta = test_generator.generate_mixture(n_tissues)
        test_mixtures.append(mixed_meth)
        test_proportions.append(true_prop)
        test_metadata.append(meta)
    
    test_file = os.path.join(output_dir, f'phase{phase}_test_mixtures.h5')
    with h5py.File(test_file, 'w') as f:
        f.create_dataset('mixed_methylation', data=np.array(test_mixtures), dtype=np.float32, compression='gzip')
        f.create_dataset('true_proportions', data=np.array(test_proportions), dtype=np.float32, compression='gzip')
        f.attrs['metadata'] = json.dumps(test_metadata)
        f.attrs['phase'] = phase
        f.attrs['n_samples'] = n_test
        f.attrs['seed'] = 43
    
    print(f"✓ Saved test set to {test_file}")

if __name__ == '__main__':
    output_dir = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/mixture_datasets'
    os.makedirs(output_dir, exist_ok=True)
    
    # Phase 1
    print("="*60)
    print("Phase 1: 2-Tissue Mixtures")
    print("="*60)
    generate_fixed_sets(phase=1, n_val=500, n_test=500, output_dir=output_dir)
    
    # Phase 2
    print("\n" + "="*60)
    print("Phase 2: 3-5 Tissue Mixtures")
    print("="*60)
    generate_fixed_sets(phase=2, n_val=1000, n_test=1000, output_dir=output_dir)
    
    # Phase 3
    print("\n" + "="*60)
    print("Phase 3: Realistic Clinical Mixtures")
    print("="*60)
    generate_fixed_sets(phase=3, n_val=2000, n_test=2000, output_dir=output_dir)
    
    print("\n" + "="*60)
    print("All mixture datasets generated successfully!")
    print("="*60)
```

**3.2 Run Generation**
```bash
cd /home/chattopa/data_storage/MethAtlas_WGBSanalysis
python3 scripts/mixture_generation/generate_mixtures.py
```

Expected time: ~30-60 minutes
Expected output: 6 HDF5 files (~2-3 GB total)

---

### Step 4: Implement Model Architecture (Days 5-6)

**4.1 Create model_deconvolution.py**

Save the TissueBERTDeconvolution class (from Section 2) to:
```
scripts/training/model_deconvolution.py
```

**4.2 Test Model Loading and Forward Pass**
```python
# scripts/training/test_model.py

import torch
import sys
sys.path.append('/home/chattopa/data_storage/MethAtlas_WGBSanalysis/scripts/training')

from model_deconvolution import TissueBERTDeconvolution, load_pretrained_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load pre-trained model
checkpoint_path = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/fullgenome_results/checkpoints/checkpoint_best_acc.pt'
model = load_pretrained_model(checkpoint_path, device)

# Test forward pass with dummy data
batch_size = 4
n_regions = 51089
seq_length = 150

dummy_methylation = torch.randint(0, 3, (batch_size, n_regions, seq_length), dtype=torch.float32).to(device)

with torch.no_grad():
    proportions = model(dummy_methylation)

print(f"Input shape: {dummy_methylation.shape}")
print(f"Output shape: {proportions.shape}")
print(f"Output range: [{proportions.min():.4f}, {proportions.max():.4f}]")
print(f"Proportions sum: {proportions.sum(dim=1)}")
print(f"All sum to 1.0: {torch.allclose(proportions.sum(dim=1), torch.ones(batch_size, device=device))}")

print("\n✓ Model test passed!")
```

Run test:
```bash
python3 scripts/training/test_model.py
```

---

### Step 5: Create Training Script (Day 7)

**5.1 Implement Training Loop**

Create comprehensive training script:
```
scripts/training/train_mixture_deconvolution.py
```

Use the training loop implementation from Section 4.

**5.2 Create Configuration Files**

Phase 1 config:
```yaml
# scripts/configs/config_phase1_2tissue.yaml

data:
  hdf5_path: '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/methylation_dataset.h5'
  metadata_csv: '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/combined_metadata.csv'
  validation_h5: '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/mixture_datasets/phase1_validation_mixtures.h5'
  test_h5: '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/mixture_datasets/phase1_test_mixtures.h5'

model:
  n_regions: 51089
  hidden_size: 512
  num_classes: 22
  dropout: 0.1
  pretrained_checkpoint: '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/fullgenome_results/checkpoints/checkpoint_best_acc.pt'

training:
  num_epochs: 30
  batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  num_workers: 12
  validation_frequency: 5
  phase: 1
  pure_sample_ratio: 0.2
  mixtures_per_epoch: 2500

optimizer:
  type: 'AdamW'
  betas: [0.9, 0.999]
  eps: 1.0e-8

scheduler:
  type: 'cosine'

loss:
  type: 'mse'

random_seed: 42

output:
  save_dir: '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase1_2tissue'
  log_every_n_steps: 50
  save_best_model: true
  save_last_model: true
```

Similar configs for Phase 2 and Phase 3 (adjust phase, mixtures_per_epoch, and checkpoint paths).

---

### Step 6: Launch Phase 1 Training (Week 2, Days 1-3)

**6.1 Create SLURM Submission Script**
```bash
# scripts/training/submit_phase1.sh

#!/bin/bash
#SBATCH --job-name=phase1_mixture
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase1_2tissue/logs/slurm_%j.out

# Activate environment
source /path/to/conda/etc/profile.d/conda.sh
conda activate your_env

# Run training
cd /home/chattopa/data_storage/MethAtlas_WGBSanalysis
python3 scripts/training/train_mixture_deconvolution.py \
    --config scripts/configs/config_phase1_2tissue.yaml

echo "Phase 1 training complete!"
```

**6.2 Submit Job**
```bash
cd /home/chattopa/data_storage/MethAtlas_WGBSanalysis
sbatch scripts/training/submit_phase1.sh
```

**6.3 Monitor Training**
```bash
# Check job status
squeue -u chattopa

# Watch output log
tail -f mixture_deconvolution_results/phase1_2tissue/logs/slurm_*.out

# TensorBoard (optional)
tensorboard --logdir=mixture_deconvolution_results/phase1_2tissue/logs --port=6006
```

---

### Step 7: Evaluate Phase 1 (Week 2, Days 6-7)

**7.1 Run Evaluation**
```python
# scripts/evaluation/evaluate_phase1.py

import torch
import h5py
import sys
sys.path.append('/home/chattopa/data_storage/MethAtlas_WGBSanalysis/scripts/training')
sys.path.append('/home/chattopa/data_storage/MethAtlas_WGBSanalysis/scripts/evaluation')

from model_deconvolution import load_pretrained_model
from evaluation_utils import comprehensive_evaluation

# Load model
checkpoint_path = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase1_2tissue/checkpoints/checkpoint_best.pt'
model = load_pretrained_model(checkpoint_path, 'cuda')

# Load test data
test_file = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/mixture_datasets/phase1_test_mixtures.h5'
with h5py.File(test_file, 'r') as f:
    test_data = {
        'mixed_methylation': f['mixed_methylation'][:],
        'true_proportions': f['true_proportions'][:]
    }

# Run evaluation
output_dir = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase1_2tissue/evaluation'
metrics = comprehensive_evaluation(model, test_data, output_dir, phase=1)

print("\nEvaluation complete! Check output directory for results.")
```

---

### Step 8: Repeat for Phases 2 and 3 (Weeks 3-4)

Follow similar process:
1. Update config to load previous phase checkpoint
2. Submit training job
3. Monitor and wait for completion (~1-2 hours per phase)
4. Run evaluation
5. Analyze results

---

## Quality Control Checkpoints

### After Each Phase:

**✓ Training Convergence**
- [ ] Loss decreasing smoothly?
- [ ] No gradient explosions (NaN/Inf)?
- [ ] Validation MAE improving?

**✓ Model Performance**
- [ ] Phase 1: MAE < 5%?
- [ ] Phase 2: MAE < 8%?
- [ ] Phase 3: MAE < 10%?

**✓ Predictions Make Sense**
- [ ] Proportions sum to 1.0?
- [ ] No negative predictions?
- [ ] Dominant tissues correctly identified?

**✓ Biological Plausibility**
- [ ] Blood-dominant mixtures predicted correctly?
- [ ] Rare tissues not over-predicted?
- [ ] Similar tissues confused appropriately?

---

## Troubleshooting Guide

### Issue 1: High Training Loss (not decreasing)

**Symptoms**: Loss stays flat or increases

**Possible causes:**
1. Learning rate too high
2. Batch size too small
3. Data loading issue

**Solutions:**
- Reduce learning rate by 10x
- Increase gradient accumulation steps
- Verify mixture generation is producing valid data

---

### Issue 2: Poor Validation Performance

**Symptoms**: Training MAE low, validation MAE high

**Possible causes:**
1. Overfitting
2. Validation set distribution mismatch
3. Pure samples not included in training

**Solutions:**
- Increase dropout from 0.1 to 0.2
- Add more data augmentation
- Ensure 20% pure samples in training

---

### Issue 3: Model Predicts All Blood

**Symptoms**: All predictions dominated by one tissue

**Possible causes:**
1. Class imbalance in training
2. Loss function issue
3. Output normalization problem

**Solutions:**
- Verify mixture proportions are diverse
- Check that proportions sum to 1.0 correctly
- Ensure MSE loss is computed correctly

---

### Issue 4: Cannot Detect Minor Components (<5%)

**Symptoms**: Small proportions predicted as 0%

**Possible causes:**
1. Insufficient training on sparse mixtures
2. Model regularization too strong
3. Loss function doesn't penalize small errors

**Solutions:**
- Add more sparse mixture examples in Phase 3
- Reduce dropout slightly
- Consider weighted MSE (higher weight for small true proportions)

---

## Success Criteria

### Phase 1 Success:
- ✓ Training completes without errors
- ✓ Overall MAE < 5%
- ✓ All tissue pairs handled (231 pairs tested)
- ✓ Correlation > 0.9 for major tissues

### Phase 2 Success:
- ✓ Overall MAE < 8%
- ✓ Handles 3, 4, and 5 tissue mixtures
- ✓ Correlation > 0.85 for major tissues

### Phase 3 Success:
- ✓ Overall MAE < 10%
- ✓ Blood proportion predicted within 5%
- ✓ Can detect >10% components with MAE < 8%
- ✓ Can detect >5% components with MAE < 12%
- ✓ Ready for PDAC clinical application

---

## Next Steps After Implementation

1. **PDAC Sample Analysis**: Apply model to 5 PDAC patients
2. **Temporal Tracking**: Monitor changes over time
3. **Clinical Correlation**: Link predictions to disease outcomes
4. **Publication**: Document methods and results
5. **Model Refinement**: Incorporate feedback from clinical data

---

**End of Implementation Plan**

**Document Version**: 1.0
**Date**: November 2024
**Authors**: Claude + User Collaboration
**Purpose**: Comprehensive guide for mixture deconvolution implementation
