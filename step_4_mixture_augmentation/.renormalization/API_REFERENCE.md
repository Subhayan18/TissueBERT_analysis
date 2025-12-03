# API Reference - Function Signatures

## Core Renormalization Functions

### 1. threshold_renormalize()
```python
def threshold_renormalize(predictions, threshold=0.05, return_mask=False):
    """
    Hard threshold renormalization - zero out predictions below threshold.
    
    Args:
        predictions: torch.Tensor or np.ndarray
                    Shape: [batch, n_tissues] or [n_tissues]
        threshold: float, default=0.05
                  Minimum proportion to keep (5%)
        return_mask: bool, default=False
                    If True, also return binary mask of kept tissues
    
    Returns:
        renormalized: Same type and shape as input
                     Proportions sum to 1.0
        mask (optional): Binary mask if return_mask=True
    
    Example:
        >>> preds = torch.tensor([0.45, 0.02, 0.48, 0.03, 0.02])
        >>> renorm = threshold_renormalize(preds, threshold=0.05)
        >>> print(renorm)  # [0.484, 0.000, 0.516, 0.000, 0.000]
    """
```

**Common Usage:**
```python
# Default (5% threshold)
renorm = threshold_renormalize(predictions)

# Custom threshold (3%)
renorm = threshold_renormalize(predictions, threshold=0.03)

# With mask
renorm, mask = threshold_renormalize(predictions, threshold=0.05, return_mask=True)
```

---

### 2. soft_threshold_renormalize()
```python
def soft_threshold_renormalize(predictions, threshold=0.05, temperature=10.0, return_weights=False):
    """
    Soft threshold with sigmoid gating - smooth suppression.
    
    Args:
        predictions: torch.Tensor or np.ndarray
                    Shape: [batch, n_tissues] or [n_tissues]
        threshold: float, default=0.05
                  Center point for sigmoid transition
        temperature: float, default=10.0
                    Sharpness of transition (higher = sharper)
        return_weights: bool, default=False
                       If True, return sigmoid weights applied
    
    Returns:
        renormalized: Same type and shape as input
                     Proportions sum to 1.0
        weights (optional): Gating weights if return_weights=True
    
    Gate function: sigmoid((predictions - threshold) × temperature)
    - predictions < threshold: weight → 0
    - predictions > threshold: weight → 1
    - predictions = threshold: weight = 0.5
    
    Example:
        >>> preds = torch.tensor([0.45, 0.02, 0.48, 0.03, 0.02])
        >>> renorm = soft_threshold_renormalize(preds, threshold=0.05, temperature=10.0)
    """
```

**Common Usage:**
```python
# Default (5% threshold, temperature 10)
renorm = soft_threshold_renormalize(predictions)

# Custom threshold
renorm = soft_threshold_renormalize(predictions, threshold=0.03)

# Sharper transition
renorm = soft_threshold_renormalize(predictions, threshold=0.05, temperature=20.0)

# Softer transition
renorm = soft_threshold_renormalize(predictions, threshold=0.05, temperature=5.0)

# With weights
renorm, weights = soft_threshold_renormalize(predictions, return_weights=True)
```

---

### 3. bayesian_sparse_renormalize()
```python
def bayesian_sparse_renormalize(predictions, prior_sparsity=0.7, confidence_threshold=0.1, return_posterior=False):
    """
    Bayesian approach with sparsity prior - probabilistic tissue presence.
    
    Args:
        predictions: torch.Tensor or np.ndarray
                    Shape: [batch, n_tissues] or [n_tissues]
        prior_sparsity: float, default=0.7
                       Expected fraction of absent tissues (70%)
        confidence_threshold: float, default=0.1
                            Minimum posterior probability to keep tissue
        return_posterior: bool, default=False
                         If True, return posterior probabilities
    
    Returns:
        renormalized: Same type and shape as input
                     Proportions sum to 1.0
        posterior (optional): Posterior probabilities if return_posterior=True
    
    Algorithm:
        1. Prior probability: P(present) = 1 - prior_sparsity
        2. Likelihood from predictions (exponential model)
        3. Posterior via Bayes rule: P(present|prediction)
        4. Keep tissues with posterior > confidence_threshold
        5. Renormalize
    
    Example:
        >>> preds = torch.tensor([0.45, 0.02, 0.48, 0.03, 0.02])
        >>> renorm = bayesian_sparse_renormalize(preds, prior_sparsity=0.7)
    """
```

**Common Usage:**
```python
# Default (70% absent)
renorm = bayesian_sparse_renormalize(predictions)

# More aggressive (80% absent)
renorm = bayesian_sparse_renormalize(predictions, prior_sparsity=0.8)

# More permissive (50% absent)
renorm = bayesian_sparse_renormalize(predictions, prior_sparsity=0.5)

# With posterior probabilities
renorm, posterior = bayesian_sparse_renormalize(predictions, return_posterior=True)
```

---

### 4. apply_renormalization()
```python
def apply_renormalization(predictions, strategy='threshold', **kwargs):
    """
    Unified interface for any renormalization strategy.
    
    Args:
        predictions: torch.Tensor or np.ndarray
                    Shape: [batch, n_tissues] or [n_tissues]
        strategy: str
                 'threshold', 'soft_threshold', 'bayesian', or 'none'
        **kwargs: Strategy-specific parameters
    
    Returns:
        renormalized: Same type and shape as input
    
    Raises:
        ValueError: If unknown strategy
    
    Example:
        >>> # Hard threshold
        >>> renorm = apply_renormalization(preds, strategy='threshold', threshold=0.05)
        >>> 
        >>> # Soft threshold
        >>> renorm = apply_renormalization(preds, strategy='soft_threshold',
        ...                               threshold=0.05, temperature=10.0)
        >>> 
        >>> # Bayesian
        >>> renorm = apply_renormalization(preds, strategy='bayesian',
        ...                               prior_sparsity=0.7)
        >>>
        >>> # No renormalization
        >>> renorm = apply_renormalization(preds, strategy='none')
    """
```

**Common Usage:**
```python
# Threshold
renorm = apply_renormalization(preds, 'threshold', threshold=0.05)

# Soft threshold
renorm = apply_renormalization(preds, 'soft_threshold', 
                               threshold=0.05, temperature=10.0)

# Bayesian
renorm = apply_renormalization(preds, 'bayesian', prior_sparsity=0.7)

# None (pass through)
renorm = apply_renormalization(preds, 'none')
```

---

### 5. compare_strategies()
```python
def compare_strategies(predictions, true_proportions=None, thresholds=[0.01, 0.03, 0.05, 0.10]):
    """
    Compare all three strategies with multiple parameters.
    
    Args:
        predictions: torch.Tensor or np.ndarray
                    Shape: [batch, n_tissues] or [n_tissues]
        true_proportions: torch.Tensor or np.ndarray, optional
                         Ground truth for computing MAE
                         Same shape as predictions
        thresholds: list of float, default=[0.01, 0.03, 0.05, 0.10]
                   Thresholds to test
    
    Returns:
        results: dict with keys:
            'strategies': dict mapping strategy_name -> {
                'predictions': renormalized predictions,
                'mae': MAE if true_proportions provided,
                'params': parameters used
            }
            'best_mae': float (if true_proportions provided)
            'best_strategy': str (if true_proportions provided)
            'best_params': dict (if true_proportions provided)
    
    Tests:
        - Hard threshold: len(thresholds) configurations
        - Soft threshold: len(thresholds) × 3 temperatures (5, 10, 20)
        - Bayesian: 3 sparsity values (0.5, 0.7, 0.8)
    
    Example:
        >>> results = compare_strategies(preds, true_props, 
        ...                             thresholds=[0.03, 0.05, 0.10])
        >>> print(f"Best: {results['best_strategy']}")
        >>> print(f"Best MAE: {results['best_mae']:.4f}")
        >>> print(f"Best params: {results['best_params']}")
        >>>
        >>> # Access specific strategy results
        >>> threshold_05 = results['strategies']['threshold_0.05']
        >>> print(f"Threshold 5%: MAE = {threshold_05['mae']:.4f}")
    """
```

**Common Usage:**
```python
# Basic comparison
results = compare_strategies(val_preds, val_true)

# Custom thresholds
results = compare_strategies(val_preds, val_true, 
                           thresholds=[0.03, 0.05, 0.07, 0.10])

# Without ground truth (no MAE computed)
results = compare_strategies(val_preds)

# Access results
best_strategy = results['best_strategy']
best_params = results['best_params']
best_mae = results['best_mae']

# Iterate over all results
for name, result in results['strategies'].items():
    print(f"{name}: MAE = {result['mae']:.4f}")
```

---

## Model Integration

### Modified forward() method
```python
class TissueBERTDeconvolution(nn.Module):
    def forward(self, methylation_status, 
               apply_renorm=None,
               renorm_strategy='threshold',
               renorm_params=None):
        """
        Forward pass with optional renormalization.
        
        Args:
            methylation_status: torch.Tensor
                               Shape: [batch, n_regions, seq_length]
                               Values: 0 (unmeth), 1 (meth), 2 (missing)
            apply_renorm: bool or None, default=None
                         If True, apply renormalization
                         If None or False, return raw normalized predictions
            renorm_strategy: str, default='threshold'
                           'threshold', 'soft_threshold', or 'bayesian'
            renorm_params: dict or None, default=None
                          Parameters for renormalization strategy
                          e.g., {'threshold': 0.05} or {'temperature': 10.0}
        
        Returns:
            proportions: torch.Tensor
                        Shape: [batch, num_classes]
                        Tissue proportions (sum to 1.0)
        
        Example:
            >>> # No renormalization (default)
            >>> preds = model(data)
            >>>
            >>> # With hard threshold
            >>> preds = model(data, 
            ...              apply_renorm=True,
            ...              renorm_strategy='threshold',
            ...              renorm_params={'threshold': 0.05})
            >>>
            >>> # With soft threshold
            >>> preds = model(data,
            ...              apply_renorm=True,
            ...              renorm_strategy='soft_threshold',
            ...              renorm_params={'threshold': 0.05, 'temperature': 10.0})
            >>>
            >>> # With Bayesian
            >>> preds = model(data,
            ...              apply_renorm=True,
            ...              renorm_strategy='bayesian',
            ...              renorm_params={'prior_sparsity': 0.7})
        """
```

**Common Usage:**
```python
model.eval()

# No renormalization (default behavior preserved)
with torch.no_grad():
    preds = model(data)

# With threshold renormalization (5%)
with torch.no_grad():
    preds = model(data, 
                 apply_renorm=True,
                 renorm_strategy='threshold',
                 renorm_params={'threshold': 0.05})

# With threshold renormalization (3%)
with torch.no_grad():
    preds = model(data,
                 apply_renorm=True, 
                 renorm_strategy='threshold',
                 renorm_params={'threshold': 0.03})

# With soft threshold
with torch.no_grad():
    preds = model(data,
                 apply_renorm=True,
                 renorm_strategy='soft_threshold',
                 renorm_params={'threshold': 0.05, 'temperature': 10.0})

# With Bayesian
with torch.no_grad():
    preds = model(data,
                 apply_renorm=True,
                 renorm_strategy='bayesian',
                 renorm_params={'prior_sparsity': 0.7})
```

---

## Parameter Recommendations

### Threshold (for threshold and soft_threshold)
| Value | Percentage | Use Case |
|-------|------------|----------|
| 0.01  | 1%        | Very permissive, keeps almost everything |
| 0.03  | 3%        | Moderate, good for Phase 2-3 |
| 0.05  | 5%        | **RECOMMENDED START** - balanced |
| 0.07  | 7%        | More aggressive |
| 0.10  | 10%       | Very aggressive, only strong predictions |

### Temperature (for soft_threshold only)
| Value | Effect |
|-------|--------|
| 5.0   | Soft transition (gradual suppression) |
| 10.0  | **RECOMMENDED** - balanced |
| 20.0  | Sharp transition (closer to hard threshold) |

### Prior Sparsity (for bayesian)
| Value | Meaning | Use Case |
|-------|---------|----------|
| 0.5   | 50% tissues absent | Low sparsity, more permissive |
| 0.7   | 70% tissues absent | **RECOMMENDED** for Phase 2 |
| 0.8   | 80% tissues absent | High sparsity, aggressive |

---

## Type Compatibility

All functions handle:
- ✅ PyTorch tensors (`torch.Tensor`)
- ✅ NumPy arrays (`np.ndarray`)
- ✅ Single samples `[n_tissues]`
- ✅ Batches `[batch, n_tissues]`
- ✅ CPU and GPU tensors

**Return type matches input type:**
```python
# NumPy in → NumPy out
np_preds = np.array([...])
np_renorm = threshold_renormalize(np_preds)  # Returns np.ndarray

# PyTorch in → PyTorch out
torch_preds = torch.tensor([...])
torch_renorm = threshold_renormalize(torch_preds)  # Returns torch.Tensor
```

---

## Import Statements

```python
# Import main model class
from model_deconvolution import TissueBERTDeconvolution

# Import renormalization functions
from model_deconvolution import (
    threshold_renormalize,
    soft_threshold_renormalize,
    bayesian_sparse_renormalize,
    apply_renormalization,
    compare_strategies
)

# Import loss function
from model_deconvolution import mixture_mse_loss

# Import model loading utility
from model_deconvolution import load_pretrained_model
```

---

## Complete Working Example

```python
import torch
from model_deconvolution import (
    load_pretrained_model,
    threshold_renormalize,
    compare_strategies
)

# Load model
model = load_pretrained_model('checkpoint_best.pt', device='cuda')
model.eval()

# Get predictions
with torch.no_grad():
    # Raw predictions
    raw_preds = model(test_data)
    
    # With renormalization (integrated)
    renorm_preds = model(test_data,
                        apply_renorm=True,
                        renorm_strategy='threshold',
                        renorm_params={'threshold': 0.05})

# Or apply renormalization separately
renorm_preds = threshold_renormalize(raw_preds, threshold=0.05)

# Compare strategies
results = compare_strategies(raw_preds, test_labels,
                           thresholds=[0.03, 0.05, 0.10])

print(f"Best strategy: {results['best_strategy']}")
print(f"Best parameters: {results['best_params']}")
print(f"Best MAE: {results['best_mae']:.4f}")

# Use best strategy
best_strategy = results['best_strategy']
best_params = results['best_params']

final_preds = model(test_data,
                   apply_renorm=True,
                   renorm_strategy=best_strategy,
                   renorm_params=best_params)
```

---

## Error Handling

All functions include:
- ✅ Input validation (shape, type)
- ✅ Edge case handling (all predictions below threshold)
- ✅ Numerical stability (avoid division by zero)
- ✅ Clear error messages

```python
# Edge case: All predictions below threshold
preds = torch.tensor([0.01, 0.02, 0.01, 0.01])  # All < 0.05
renorm = threshold_renormalize(preds, threshold=0.05)
# Handled gracefully: returns uniform distribution [0.25, 0.25, 0.25, 0.25]

# Edge case: Single very high prediction
preds = torch.tensor([0.95, 0.01, 0.02, 0.01, 0.01])
renorm = threshold_renormalize(preds, threshold=0.05)
# Result: [1.0, 0.0, 0.0, 0.0, 0.0] - mass consolidated
```

---

## Performance Notes

| Function | Complexity | Memory | GPU Compatible |
|----------|------------|--------|----------------|
| threshold_renormalize | O(n) | Minimal | ✅ Yes |
| soft_threshold_renormalize | O(n) | Minimal | ✅ Yes |
| bayesian_sparse_renormalize | O(n) | Minimal | ✅ Yes |
| compare_strategies | O(k×n) | Moderate | ✅ Yes |

Where:
- n = number of tissues (22)
- k = number of configurations tested (~20)

All functions are fast enough for real-time inference!

---

## Version Information

**API Version:** 1.0  
**Compatibility:** Python 3.7+, PyTorch 1.8+, NumPy 1.19+  
**Last Updated:** December 2024

---

This API reference provides all function signatures, parameters, and usage examples. For more context and examples, see the other documentation files!
