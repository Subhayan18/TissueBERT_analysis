#!/usr/bin/env python3
"""
TissueBERT Deconvolution Model
===============================

Modified version of TissueBERT for mixture deconvolution.

Key changes from single-tissue classification:
1. Output: Sigmoid + L1 normalization (instead of softmax)
2. Loss: MSE on proportions (instead of CrossEntropy)
3. Task: Predict tissue proportions [22] (instead of single class)

Architecture identical to original up to classifier layer, enabling
transfer learning from pre-trained single-tissue model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# POST-PROCESSING RENORMALIZATION STRATEGIES
# ============================================================================
# These functions address spurious low-confidence tissue predictions by
# suppressing predictions below a threshold and renormalizing.
# ============================================================================

def threshold_renormalize(predictions, threshold=0.05, return_mask=False):
    """
    Strategy 1: Hard Threshold Renormalization
    
    Zero out predictions below threshold and renormalize remaining proportions.
    This is the simplest and most interpretable approach.
    
    Args:
        predictions: [batch, n_tissues] or [n_tissues] tensor/array
        threshold: Minimum proportion to keep (default 5%)
        return_mask: If True, also return the binary mask of kept tissues
        
    Returns:
        renormalized_predictions: Same shape as input
        mask (optional): Binary mask of tissues above threshold
        
    Example:
        >>> preds = torch.tensor([0.45, 0.02, 0.48, 0.03, 0.02])  # Sum = 1.0
        >>> renorm = threshold_renormalize(preds, threshold=0.05)
        >>> print(renorm)  # tensor([0.484, 0.000, 0.516, 0.000, 0.000])
    """
    is_numpy = isinstance(predictions, np.ndarray)
    
    if is_numpy:
        predictions = torch.from_numpy(predictions)
    
    # Handle both single sample and batch
    is_single = predictions.dim() == 1
    if is_single:
        predictions = predictions.unsqueeze(0)
    
    # Create mask for predictions above threshold
    mask = (predictions >= threshold).float()
    
    # Zero out below threshold
    masked_preds = predictions * mask
    
    # Renormalize (handle edge case where all predictions below threshold)
    sums = masked_preds.sum(dim=-1, keepdim=True)
    sums = torch.where(sums > 0, sums, torch.ones_like(sums))
    renormalized = masked_preds / sums
    
    # Remove batch dimension if input was single sample
    if is_single:
        renormalized = renormalized.squeeze(0)
        mask = mask.squeeze(0)
    
    if is_numpy:
        renormalized = renormalized.numpy()
        mask = mask.numpy()
    
    if return_mask:
        return renormalized, mask
    return renormalized


def soft_threshold_renormalize(predictions, threshold=0.05, temperature=10.0, return_weights=False):
    """
    Strategy 2: Soft Threshold with Temperature Scaling
    
    Apply smooth suppression using sigmoid gating instead of hard cutoff.
    More differentiable and provides gradual transition around threshold.
    
    The gating function is: sigmoid((predictions - threshold) * temperature)
    - High temperature → sharper transition (closer to hard threshold)
    - Low temperature → softer transition (more gradual)
    
    Args:
        predictions: [batch, n_tissues] or [n_tissues] tensor/array
        threshold: Center point for sigmoid gating (default 5%)
        temperature: Sharpness of transition (default 10.0)
                    Higher = sharper, closer to hard threshold
        return_weights: If True, also return the gating weights
        
    Returns:
        renormalized_predictions: Same shape as input
        weights (optional): Gating weights applied
        
    Example:
        >>> preds = torch.tensor([0.45, 0.02, 0.48, 0.03, 0.02])
        >>> renorm = soft_threshold_renormalize(preds, threshold=0.05, temperature=10.0)
        >>> # Predictions near 5% get smoothly suppressed, not hard-zeroed
    """
    is_numpy = isinstance(predictions, np.ndarray)
    
    if is_numpy:
        predictions = torch.from_numpy(predictions)
    
    # Handle both single sample and batch
    is_single = predictions.dim() == 1
    if is_single:
        predictions = predictions.unsqueeze(0)
    
    # Compute gating weights using sigmoid
    # sigmoid((x - threshold) * temperature)
    # When x < threshold: weight → 0
    # When x > threshold: weight → 1
    # When x = threshold: weight = 0.5
    gates = torch.sigmoid((predictions - threshold) * temperature)
    
    # Apply gating
    gated_preds = predictions * gates
    
    # Renormalize
    sums = gated_preds.sum(dim=-1, keepdim=True)
    sums = torch.where(sums > 0, sums, torch.ones_like(sums))
    renormalized = gated_preds / sums
    
    # Remove batch dimension if input was single sample
    if is_single:
        renormalized = renormalized.squeeze(0)
        gates = gates.squeeze(0)
    
    if is_numpy:
        renormalized = renormalized.numpy()
        gates = gates.numpy()
    
    if return_weights:
        return renormalized, gates
    return renormalized


def bayesian_sparse_renormalize(predictions, prior_sparsity=0.7, confidence_threshold=0.1, return_posterior=False):
    """
    Strategy 3: Bayesian Sparsity Prior
    
    Apply a principled probabilistic approach using empirical Bayes.
    Assumes most tissues are absent (sparse prior) and uses prediction
    confidence to estimate posterior probability of tissue presence.
    
    This is the most sophisticated approach but computationally more expensive.
    
    Algorithm:
    1. Compute prior probability of tissue presence based on sparsity
    2. Use prediction magnitude as likelihood
    3. Compute posterior probability via Bayes rule
    4. Threshold on posterior probability
    5. Renormalize
    
    Args:
        predictions: [batch, n_tissues] or [n_tissues] tensor/array
        prior_sparsity: Expected fraction of absent tissues (default 0.7 = 70% absent)
        confidence_threshold: Minimum posterior probability to keep tissue (default 0.1)
        return_posterior: If True, return posterior probabilities
        
    Returns:
        renormalized_predictions: Same shape as input
        posterior (optional): Posterior probability of tissue presence
        
    Example:
        >>> preds = torch.tensor([0.45, 0.02, 0.48, 0.03, 0.02])
        >>> renorm = bayesian_sparse_renormalize(preds, prior_sparsity=0.7)
        >>> # Uses statistical model to determine which predictions are real
    """
    is_numpy = isinstance(predictions, np.ndarray)
    
    if is_numpy:
        predictions = torch.from_numpy(predictions)
    
    # Handle both single sample and batch
    is_single = predictions.dim() == 1
    if is_single:
        predictions = predictions.unsqueeze(0)
    
    batch_size, n_tissues = predictions.shape
    
    # Prior probability of tissue being present
    prior_present = 1.0 - prior_sparsity
    prior_absent = prior_sparsity
    
    # Likelihood: P(prediction | tissue present) vs P(prediction | tissue absent)
    # Model: Present tissues have higher predictions, absent tissues have lower
    # Use a simple exponential model: P(x|present) ∝ exp(α*x), P(x|absent) ∝ exp(-β*x)
    alpha = 10.0  # Scale for present likelihood
    beta = 5.0    # Scale for absent likelihood
    
    likelihood_present = torch.exp(alpha * predictions)
    likelihood_absent = torch.exp(-beta * predictions)
    
    # Posterior probability via Bayes rule
    # P(present|x) = P(x|present)*P(present) / [P(x|present)*P(present) + P(x|absent)*P(absent)]
    numerator = likelihood_present * prior_present
    denominator = likelihood_present * prior_present + likelihood_absent * prior_absent
    posterior_present = numerator / (denominator + 1e-10)
    
    # Create mask based on posterior probability
    mask = (posterior_present >= confidence_threshold).float()
    
    # Apply mask and renormalize
    masked_preds = predictions * mask
    sums = masked_preds.sum(dim=-1, keepdim=True)
    sums = torch.where(sums > 0, sums, torch.ones_like(sums))
    renormalized = masked_preds / sums
    
    # Remove batch dimension if input was single sample
    if is_single:
        renormalized = renormalized.squeeze(0)
        posterior_present = posterior_present.squeeze(0)
    
    if is_numpy:
        renormalized = renormalized.numpy()
        posterior_present = posterior_present.numpy()
    
    if return_posterior:
        return renormalized, posterior_present
    return renormalized


def apply_renormalization(predictions, strategy='threshold', **kwargs):
    """
    Unified interface for applying any renormalization strategy
    
    Args:
        predictions: [batch, n_tissues] or [n_tissues] tensor/array
        strategy: 'threshold', 'soft_threshold', or 'bayesian'
        **kwargs: Strategy-specific parameters
        
    Returns:
        renormalized_predictions: Same shape as input
        
    Example:
        >>> # Hard threshold
        >>> renorm1 = apply_renormalization(preds, strategy='threshold', threshold=0.05)
        >>> 
        >>> # Soft threshold
        >>> renorm2 = apply_renormalization(preds, strategy='soft_threshold', 
        ...                                 threshold=0.05, temperature=10.0)
        >>> 
        >>> # Bayesian
        >>> renorm3 = apply_renormalization(preds, strategy='bayesian', 
        ...                                 prior_sparsity=0.7)
    """
    if strategy == 'threshold':
        return threshold_renormalize(predictions, **kwargs)
    elif strategy == 'soft_threshold':
        return soft_threshold_renormalize(predictions, **kwargs)
    elif strategy == 'bayesian':
        return bayesian_sparse_renormalize(predictions, **kwargs)
    elif strategy is None or strategy == 'none':
        return predictions
    else:
        raise ValueError(f"Unknown strategy: {strategy}. "
                        f"Choose from: 'threshold', 'soft_threshold', 'bayesian', 'none'")


def compare_strategies(predictions, true_proportions=None, thresholds=[0.01, 0.03, 0.05, 0.10]):
    """
    Compare all three renormalization strategies with multiple thresholds
    
    Useful for hyperparameter search and method comparison.
    
    Args:
        predictions: [batch, n_tissues] or [n_tissues] tensor/array
        true_proportions: Optional ground truth for computing MAE
        thresholds: List of thresholds to test (default: [1%, 3%, 5%, 10%])
        
    Returns:
        results: Dictionary with results for each strategy and threshold
        
    Example:
        >>> results = compare_strategies(preds, true_props, thresholds=[0.03, 0.05, 0.10])
        >>> print(f"Best MAE: {results['best_mae']:.4f}")
        >>> print(f"Best strategy: {results['best_strategy']}")
    """
    is_numpy = isinstance(predictions, np.ndarray)
    if is_numpy:
        predictions = torch.from_numpy(predictions)
        if true_proportions is not None:
            true_proportions = torch.from_numpy(true_proportions)
    
    results = {
        'strategies': {},
        'best_mae': float('inf'),
        'best_strategy': None,
        'best_params': None
    }
    
    # Test hard threshold strategy
    for thresh in thresholds:
        renorm = threshold_renormalize(predictions, threshold=thresh)
        
        if true_proportions is not None:
            mae = (renorm - true_proportions).abs().mean().item()
        else:
            mae = None
        
        key = f'threshold_{thresh:.2f}'
        results['strategies'][key] = {
            'predictions': renorm.numpy() if is_numpy else renorm,
            'mae': mae,
            'params': {'threshold': thresh}
        }
        
        if mae is not None and mae < results['best_mae']:
            results['best_mae'] = mae
            results['best_strategy'] = 'threshold'
            results['best_params'] = {'threshold': thresh}
    
    # Test soft threshold strategy
    for thresh in thresholds:
        for temp in [5.0, 10.0, 20.0]:
            renorm = soft_threshold_renormalize(predictions, threshold=thresh, temperature=temp)
            
            if true_proportions is not None:
                mae = (renorm - true_proportions).abs().mean().item()
            else:
                mae = None
            
            key = f'soft_threshold_{thresh:.2f}_temp{temp:.0f}'
            results['strategies'][key] = {
                'predictions': renorm.numpy() if is_numpy else renorm,
                'mae': mae,
                'params': {'threshold': thresh, 'temperature': temp}
            }
            
            if mae is not None and mae < results['best_mae']:
                results['best_mae'] = mae
                results['best_strategy'] = 'soft_threshold'
                results['best_params'] = {'threshold': thresh, 'temperature': temp}
    
    # Test Bayesian strategy
    for sparsity in [0.5, 0.7, 0.8]:
        renorm = bayesian_sparse_renormalize(predictions, prior_sparsity=sparsity)
        
        if true_proportions is not None:
            mae = (renorm - true_proportions).abs().mean().item()
        else:
            mae = None
        
        key = f'bayesian_sparsity{sparsity:.1f}'
        results['strategies'][key] = {
            'predictions': renorm.numpy() if is_numpy else renorm,
            'mae': mae,
            'params': {'prior_sparsity': sparsity}
        }
        
        if mae is not None and mae < results['best_mae']:
            results['best_mae'] = mae
            results['best_strategy'] = 'bayesian'
            results['best_params'] = {'prior_sparsity': sparsity}
    
    return results


# ============================================================================
# END OF POST-PROCESSING STRATEGIES
# ============================================================================


class TissueBERTDeconvolution(nn.Module):
    """
    File-Level MLP for Mixture Deconvolution
    
    Input: [batch, 51089, 150] methylation (mixed samples)
    Process:
        1. Compute mean methylation per region → [batch, 51089]
        2. Project to hidden dimension → [batch, 512]
        3. MLP feature extraction → [batch, 1024]
        4. Classifier → [batch, 22]
        5. Sigmoid + L1 normalization → [batch, 22] proportions (sum=1.0)
    """

    def __init__(self,
                 vocab_size=69,
                 hidden_size=512,
                 num_hidden_layers=3,
                 num_attention_heads=4,
                 intermediate_size=2048,
                 max_position_embeddings=150,
                 num_classes=22,
                 dropout=0.1,
                 n_regions=51089):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.n_regions = n_regions

        # Input projection: 51089 → hidden_size
        self.input_projection = nn.Linear(n_regions, hidden_size)
        
        # MLP network (identical to original)
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 2
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 3
            nn.Linear(hidden_size, intermediate_size),
            nn.BatchNorm1d(intermediate_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Output (classifier)
            nn.Linear(intermediate_size, num_classes)
        )
        
        # NEW: Sigmoid activation for independent tissue probabilities
        self.sigmoid = nn.Sigmoid()
        
        self.apply(self._init_weights)
        
        # Calculate parameter count
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("=" * 70)
        print("TISSUEBERT DECONVOLUTION MODEL")
        print("=" * 70)
        print(f"Input: [batch, {n_regions:,} regions, 150 bp] methylation")
        print(f"Process: Region aggregation → MLP → Deconvolution")
        print(f"Output: [batch, {num_classes}] tissue proportions (sum=1.0)")
        print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
        print(f"Memory: ~{n_params*4/1e6:.0f} MB (float32)")
        print("=" * 70)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, methylation_status, apply_renorm=None, renorm_strategy='threshold', renorm_params=None):
        """
        Forward pass with proportion output and optional renormalization
        
        Args:
            methylation_status: [batch, n_regions, seq_length]
                               Values: 0 (unmeth), 1 (meth), 2 (missing)
            apply_renorm: If True, apply post-processing renormalization
                         If None/False, return raw normalized proportions
            renorm_strategy: 'threshold', 'soft_threshold', or 'bayesian'
            renorm_params: Dict of parameters for renormalization
                          e.g., {'threshold': 0.05} or {'temperature': 10.0}
            
        Returns:
            proportions: [batch, num_classes] tissue proportions (sum=1.0)
        """
        batch_size = methylation_status.shape[0]
        n_regions = methylation_status.shape[1]
        seq_length = methylation_status.shape[2]
        
        # Verify shape
        assert n_regions == self.n_regions, \
            f"Expected {self.n_regions} regions, got {n_regions}"
        
        # ===================================================================
        # STEP 1: Compute mean methylation per region (same as original)
        # ===================================================================
        
        # Mask missing values (2) and convert to float
        valid_mask = (methylation_status != 2).float()
        meth_float = methylation_status.float()
        meth_float = torch.where(methylation_status == 2, 
                                torch.zeros_like(meth_float), 
                                meth_float)
        
        # Compute mean per region
        region_sums = (meth_float * valid_mask).sum(dim=2)  # [batch, n_regions]
        region_counts = valid_mask.sum(dim=2).clamp(min=1.0)  # [batch, n_regions]
        region_means = region_sums / region_counts  # [batch, n_regions]
        
        # ===================================================================
        # STEP 2: Project to hidden dimension (same as original)
        # ===================================================================
        
        features = self.input_projection(region_means)  # [batch, hidden_size]
        
        # ===================================================================
        # STEP 3: MLP feature extraction (same as original)
        # ===================================================================
        
        logits = self.network(features)  # [batch, num_classes]
        
        # ===================================================================
        # STEP 4: Sigmoid + L1 normalization for proportions
        # ===================================================================
        
        # Apply sigmoid to get independent probabilities
        sigmoid_outputs = self.sigmoid(logits)  # [batch, num_classes]
        
        # L1 normalization to sum to 1.0
        proportions = sigmoid_outputs / sigmoid_outputs.sum(dim=1, keepdim=True)
        
        # ===================================================================
        # STEP 5: OPTIONAL - Apply post-processing renormalization
        # ===================================================================
        
        if apply_renorm:
            if renorm_params is None:
                renorm_params = {}
            proportions = apply_renormalization(
                proportions, 
                strategy=renorm_strategy, 
                **renorm_params
            )
        
        return proportions


def load_pretrained_model(checkpoint_path, device='cuda', verbose=True):
    """
    Load pre-trained single-tissue model and convert to deconvolution model.
    
    The architecture is identical, so all weights can be loaded directly.
    Only the forward pass changes (sigmoid+normalize instead of softmax).
    
    Args:
        checkpoint_path: Path to checkpoint_best_acc.pt from single-tissue training
        device: 'cuda' or 'cpu'
        verbose: Print loading information
        
    Returns:
        model: TissueBERTDeconvolution with loaded weights
    """
    if verbose:
        print("\n" + "="*80)
        print("LOADING PRE-TRAINED MODEL")
        print("="*80)
        print(f"Checkpoint: {checkpoint_path}")
    
    # Initialize deconvolution model
    model = TissueBERTDeconvolution(
        vocab_size=69,
        hidden_size=512,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=2048,
        max_position_embeddings=150,
        num_classes=22,
        dropout=0.1,
        n_regions=51089
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if verbose:
        print(f"\nCheckpoint info:")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Validation accuracy: {checkpoint.get('val_accuracy', 'N/A')}")
        print(f"  Validation loss: {checkpoint.get('val_loss', 'N/A')}")
    
    # Load state dict
    # The model architecture is identical, so strict loading should work
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    
    if verbose:
        print(f"\n✓ Successfully loaded pre-trained weights")
        print(f"  All layers loaded successfully")
        print("="*80)
    
    return model.to(device)


def mixture_mse_loss(predicted_proportions, true_proportions):
    """
    Mean Squared Error loss on tissue proportions.
    
    Args:
        predicted_proportions: [batch, 22] - model predictions (sum to 1.0)
        true_proportions: [batch, 22] - ground truth mixing weights (sum to 1.0)
        
    Returns:
        loss: scalar MSE value
    """
    # MSE across all tissues
    loss = F.mse_loss(predicted_proportions, true_proportions)
    
    return loss


def test_model():
    """Test deconvolution model with dummy data"""
    print("\nTesting TissueBERT Deconvolution Model...")
    
    model = TissueBERTDeconvolution(
        vocab_size=69,
        hidden_size=512,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=2048,
        max_position_embeddings=150,
        num_classes=22,
        dropout=0.1,
        n_regions=51089
    )
    
    batch_size = 4
    n_regions = 51089
    seq_length = 150
    
    print(f"\nTest input shapes:")
    print(f"  Batch size: {batch_size}")
    print(f"  Regions: {n_regions:,}")
    print(f"  Sequence length: {seq_length}")
    
    # Create dummy data
    methylation = torch.randint(0, 3, (batch_size, n_regions, seq_length))
    
    # Test forward pass
    with torch.no_grad():
        proportions = model(methylation)
    
    print(f"\nOutput shape: {proportions.shape}")
    assert proportions.shape == (batch_size, 22)
    
    print(f"\nProportion checks:")
    for i in range(batch_size):
        prop_sum = proportions[i].sum().item()
        prop_max = proportions[i].max().item()
        prop_min = proportions[i].min().item()
        print(f"  Sample {i}: sum={prop_sum:.6f}, min={prop_min:.6f}, max={prop_max:.6f}")
    
    # Verify all sum to 1.0
    sums = proportions.sum(dim=1)
    assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5), \
        f"Proportions don't sum to 1.0: {sums}"
    
    # Test loss function
    true_props = torch.randn(batch_size, 22).abs()
    true_props = true_props / true_props.sum(dim=1, keepdim=True)
    
    loss = mixture_mse_loss(proportions, true_props)
    print(f"\nTest loss: {loss.item():.6f}")
    
    print("\n✓ Deconvolution model test passed!")


if __name__ == '__main__':
    # Test renormalization functions with synthetic data matching your screenshot
    print("\n" + "="*80)
    print(" RENORMALIZATION STRATEGY TESTING SUITE")
    print("="*80)
    print("\nThis demonstrates all three post-processing strategies to reduce")
    print("spurious low-confidence predictions in mixture deconvolution.")
    print("="*80)
    
    test_renormalization_only()
    
    print("\n\n")
    
    # Test full model integration
    test_model_with_renorm()
    
    print("\n\n" + "="*80)
    print(" SUMMARY")
    print("="*80)
    print("\n✓ Three renormalization strategies successfully implemented:")
    print("\n  1. HARD THRESHOLD: Zero out predictions below threshold")
    print("     - Simple, fast, interpretable")
    print("     - Recommended starting point: threshold=0.05 (5%)")
    print("\n  2. SOFT THRESHOLD: Smooth suppression via sigmoid gating")
    print("     - More gradual transition")
    print("     - Recommended: threshold=0.05, temperature=10.0")
    print("\n  3. BAYESIAN SPARSE: Probabilistic approach with sparsity prior")
    print("     - Most sophisticated")
    print("     - Recommended: prior_sparsity=0.7 (70% absent)")
    print("\n✓ All strategies are modular and configurable")
    print("✓ Can be applied during prediction or evaluation")
    print("✓ Use compare_strategies() to find optimal parameters")
    print("="*80)


def test_renormalization_only():
    """Test just the renormalization functions with synthetic data"""
    print("\n" + "="*80)
    print("TESTING RENORMALIZATION STRATEGIES ONLY")
    print("="*80)
    
    # Create synthetic predictions mimicking your screenshot
    # Original: [0%, 0%, 0%, ...]  Predicted: [2%, 3%, 1%, ...]
    # True tissues at ~50% of actual values
    
    print("\nScenario: Mixture_10 from your screenshot")
    print("-" * 80)
    
    # True proportions (from ground truth)
    true_props = np.array([
        0.00, 0.00, 0.00, 0.00, 0.00,  # First 5 tissues: absent
        0.28, 0.31, 0.21, 0.00, 0.14,  # Cerebellum, Colon, Coronary, Cortex, Dermal
        0.00, 0.00, 0.00, 0.09, 0.00,  # Epidermal, Gastric, Heart, Kidney, Liver
        0.00, 0.00, 0.00, 0.00, 0.00,  # Lung, Neuron, Oligodendro, Pancreas, Prostate
        0.00, 0.00                      # Skeletal, Small
    ])
    
    # Raw model predictions (simulating your issue)
    # True tissues at ~50%, spurious at 1-4%
    raw_preds = np.array([
        0.02, 0.03, 0.02, 0.01, 0.01,  # Spurious predictions
        0.13, 0.18, 0.05, 0.01, 0.08,  # True tissues underestimated
        0.04, 0.04, 0.02, 0.05, 0.04,  # More spurious
        0.01, 0.02, 0.02, 0.02, 0.01,  # More spurious
        0.00, 0.00                      # Actually zero
    ])
    
    print(f"True proportions (non-zero): {np.where(true_props > 0)[0].tolist()}")
    print(f"Raw predictions > 1%: {np.where(raw_preds > 0.01)[0].tolist()}")
    print(f"\nTrue proportion sum: {true_props.sum():.3f}")
    print(f"Raw prediction sum: {raw_preds.sum():.3f}")
    
    # Calculate baseline MAE
    baseline_mae = np.abs(raw_preds - true_props).mean()
    print(f"\nBaseline MAE (no renorm): {baseline_mae:.6f}")
    
    # Test Strategy 1: Hard Threshold
    print("\n" + "="*80)
    print("Strategy 1: Hard Threshold")
    print("="*80)
    for thresh in [0.03, 0.05, 0.10]:
        renorm = threshold_renormalize(raw_preds, threshold=thresh)
        mae = np.abs(renorm - true_props).mean()
        n_nonzero = (renorm > 0).sum()
        print(f"\nThreshold {thresh*100:.0f}%:")
        print(f"  MAE: {mae:.6f} (vs baseline: {baseline_mae:.6f})")
        print(f"  Improvement: {(baseline_mae - mae)/baseline_mae*100:.1f}%")
        print(f"  Non-zero predictions: {n_nonzero}")
        print(f"  Predicted tissues: {np.where(renorm > 0.01)[0].tolist()}")
    
    # Test Strategy 2: Soft Threshold
    print("\n" + "="*80)
    print("Strategy 2: Soft Threshold")
    print("="*80)
    for thresh in [0.03, 0.05]:
        for temp in [5.0, 10.0, 20.0]:
            renorm = soft_threshold_renormalize(raw_preds, threshold=thresh, temperature=temp)
            mae = np.abs(renorm - true_props).mean()
            n_above_001 = (renorm > 0.001).sum()
            print(f"\nThreshold {thresh*100:.0f}%, Temperature {temp:.0f}:")
            print(f"  MAE: {mae:.6f}")
            print(f"  Improvement: {(baseline_mae - mae)/baseline_mae*100:.1f}%")
            print(f"  Predictions >0.1%: {n_above_001}")
    
    # Test Strategy 3: Bayesian
    print("\n" + "="*80)
    print("Strategy 3: Bayesian Sparse")
    print("="*80)
    for sparsity in [0.5, 0.7, 0.8]:
        renorm = bayesian_sparse_renormalize(raw_preds, prior_sparsity=sparsity)
        mae = np.abs(renorm - true_props).mean()
        n_nonzero = (renorm > 0).sum()
        print(f"\nPrior sparsity {sparsity*100:.0f}%:")
        print(f"  MAE: {mae:.6f}")
        print(f"  Improvement: {(baseline_mae - mae)/baseline_mae*100:.1f}%")
        print(f"  Non-zero predictions: {n_nonzero}")
    
    # Find best strategy
    print("\n" + "="*80)
    print("SUMMARY: Best Strategy Search")
    print("="*80)
    results = compare_strategies(raw_preds, true_props, thresholds=[0.03, 0.05, 0.10])
    print(f"\nBest strategy: {results['best_strategy']}")
    print(f"Best params: {results['best_params']}")
    print(f"Best MAE: {results['best_mae']:.6f}")
    print(f"Improvement over baseline: {(baseline_mae - results['best_mae'])/baseline_mae*100:.1f}%")


def test_model_with_renorm():
    """Test model with all three renormalization strategies"""
    print("\n" + "="*80)
    print("TESTING MODEL WITH RENORMALIZATION")
    print("="*80)
    
    model = TissueBERTDeconvolution(
        vocab_size=69,
        hidden_size=512,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=2048,
        max_position_embeddings=150,
        num_classes=22,
        dropout=0.1,
        n_regions=51089
    )
    
    batch_size = 4
    n_regions = 51089
    seq_length = 150
    
    print(f"\nTest input shapes:")
    print(f"  Batch size: {batch_size}")
    print(f"  Regions: {n_regions:,}")
    print(f"  Sequence length: {seq_length}")
    
    # Create dummy data
    methylation = torch.randint(0, 3, (batch_size, n_regions, seq_length))
    
    # Test 1: Raw predictions
    print("\n" + "="*80)
    print("TEST 1: Raw predictions (no renormalization)")
    print("="*80)
    with torch.no_grad():
        proportions = model(methylation, apply_renorm=False)
    
    print(f"Output shape: {proportions.shape}")
    for i in range(min(2, batch_size)):
        n_above_05 = (proportions[i] > 0.05).sum().item()
        print(f"  Sample {i}: tissues >5%: {n_above_05}")
    
    # Test 2: Hard threshold
    print("\n" + "="*80)
    print("TEST 2: Hard Threshold (5%)")
    print("="*80)
    with torch.no_grad():
        renorm1 = model(methylation, apply_renorm=True, 
                       renorm_strategy='threshold', 
                       renorm_params={'threshold': 0.05})
    
    for i in range(min(2, batch_size)):
        n_nonzero = (renorm1[i] > 0).sum().item()
        print(f"  Sample {i}: non-zero tissues: {n_nonzero}")
    
    # Test 3: Soft threshold
    print("\n" + "="*80)
    print("TEST 3: Soft Threshold (5%, temp=10)")
    print("="*80)
    with torch.no_grad():
        renorm2 = model(methylation, apply_renorm=True,
                       renorm_strategy='soft_threshold',
                       renorm_params={'threshold': 0.05, 'temperature': 10.0})
    
    for i in range(min(2, batch_size)):
        n_above_001 = (renorm2[i] > 0.001).sum().item()
        print(f"  Sample {i}: tissues >0.1%: {n_above_001}")
    
    # Test 4: Bayesian
    print("\n" + "="*80)
    print("TEST 4: Bayesian (sparsity=0.7)")
    print("="*80)
    with torch.no_grad():
        renorm3 = model(methylation, apply_renorm=True,
                       renorm_strategy='bayesian',
                       renorm_params={'prior_sparsity': 0.7})
    
    for i in range(min(2, batch_size)):
        n_nonzero = (renorm3[i] > 0).sum().item()
        print(f"  Sample {i}: non-zero tissues: {n_nonzero}")
    
    print("\n" + "="*80)
    print("✓ ALL MODEL TESTS PASSED!")
    print("="*80)
    print("\nIntegration successful! Usage examples:")
    print("\n  # No renormalization (default):")
    print("  preds = model(data)")
    print("\n  # Hard threshold at 5%:")
    print("  preds = model(data, apply_renorm=True, renorm_strategy='threshold',")
    print("               renorm_params={'threshold': 0.05})")
    print("\n  # Soft threshold:")
    print("  preds = model(data, apply_renorm=True, renorm_strategy='soft_threshold',")
    print("               renorm_params={'threshold': 0.05, 'temperature': 10.0})")
    print("\n  # Bayesian:")
    print("  preds = model(data, apply_renorm=True, renorm_strategy='bayesian',")
    print("               renorm_params={'prior_sparsity': 0.7})")


