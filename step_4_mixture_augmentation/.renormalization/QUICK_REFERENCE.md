# Renormalization Quick Reference

## TL;DR - Copy-Paste Examples

### Basic Usage: Hard Threshold (Start Here)

```python
from model_deconvolution_updated import TissueBERTDeconvolution, load_pretrained_model

# Load model
model = load_pretrained_model(checkpoint_path, device='cuda')
model.eval()

# Option 1: Apply during prediction
with torch.no_grad():
    predictions = model(methylation_data,
                       apply_renorm=True,
                       renorm_strategy='threshold',
                       renorm_params={'threshold': 0.05})

# Option 2: Apply after prediction
with torch.no_grad():
    raw_predictions = model(methylation_data)
    
from model_deconvolution_updated import threshold_renormalize
predictions = threshold_renormalize(raw_predictions, threshold=0.05)
```

---

## Find Best Threshold (10 minutes)

```python
from model_deconvolution_updated import threshold_renormalize

# Get predictions on validation set
model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for batch in val_loader:
        preds = model(batch['methylation'])
        all_preds.append(preds)
        all_true.append(batch['proportions'])

all_preds = torch.cat(all_preds, dim=0)
all_true = torch.cat(all_true, dim=0)

# Test different thresholds
print("Threshold | MAE      | Improvement")
print("-" * 40)

baseline_mae = (all_preds - all_true).abs().mean().item()
print(f"None      | {baseline_mae:.6f} | baseline")

for thresh in [0.01, 0.03, 0.05, 0.10]:
    renorm = threshold_renormalize(all_preds, threshold=thresh)
    mae = (renorm - all_true).abs().mean().item()
    improvement = (baseline_mae - mae) / baseline_mae * 100
    print(f"{thresh:.2f}     | {mae:.6f} | {improvement:+.1f}%")
```

---

## Compare All Three Strategies (30 minutes)

```python
from model_deconvolution_updated import (
    threshold_renormalize,
    soft_threshold_renormalize,
    bayesian_sparse_renormalize,
    compare_strategies
)

# Get predictions
with torch.no_grad():
    raw_preds = model(val_data)
    true_props = val_labels

# Automatic comparison
results = compare_strategies(
    raw_preds,
    true_props,
    thresholds=[0.03, 0.05, 0.10]
)

print(f"\nBest Strategy: {results['best_strategy']}")
print(f"Best Parameters: {results['best_params']}")
print(f"Best MAE: {results['best_mae']:.6f}")

# Manual comparison (more control)
print("\nManual Comparison:")
print("Strategy              | MAE      | Improvement")
print("-" * 50)

# Baseline
baseline_mae = (raw_preds - true_props).abs().mean().item()
print(f"No renormalization    | {baseline_mae:.6f} | baseline")

# Hard threshold
for thresh in [0.03, 0.05, 0.10]:
    renorm = threshold_renormalize(raw_preds, threshold=thresh)
    mae = (renorm - true_props).abs().mean().item()
    improvement = (baseline_mae - mae) / baseline_mae * 100
    print(f"Hard Threshold {thresh:.2f}  | {mae:.6f} | {improvement:+.1f}%")

# Soft threshold
for thresh in [0.05]:
    for temp in [5.0, 10.0, 20.0]:
        renorm = soft_threshold_renormalize(raw_preds, 
                                           threshold=thresh, 
                                           temperature=temp)
        mae = (renorm - true_props).abs().mean().item()
        improvement = (baseline_mae - mae) / baseline_mae * 100
        print(f"Soft Thresh {thresh:.2f} T{temp:>4.0f} | {mae:.6f} | {improvement:+.1f}%")

# Bayesian
for sparsity in [0.5, 0.7, 0.8]:
    renorm = bayesian_sparse_renormalize(raw_preds, 
                                        prior_sparsity=sparsity)
    mae = (renorm - true_props).abs().mean().item()
    improvement = (baseline_mae - mae) / baseline_mae * 100
    print(f"Bayesian Sparse {sparsity:.1f}  | {mae:.6f} | {improvement:+.1f}%")
```

---

## Integration with Your Training Script

### Add to Evaluation Function

```python
# In train_deconvolution.py

def evaluate_with_renorm(model, dataloader, device, 
                        apply_renorm=False, 
                        renorm_strategy='threshold',
                        threshold=0.05):
    """
    Evaluate model with optional renormalization.
    
    Args:
        model: Deconvolution model
        dataloader: Validation or test DataLoader
        device: 'cuda' or 'cpu'
        apply_renorm: Whether to apply renormalization
        renorm_strategy: 'threshold', 'soft_threshold', or 'bayesian'
        threshold: Threshold value (for threshold strategies)
    
    Returns:
        metrics: Dict with MAE, correlation, etc.
    """
    model.eval()
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for batch in dataloader:
            methylation = batch['methylation'].to(device)
            true_props = batch['proportions'].to(device)
            
            # Get predictions
            if apply_renorm:
                if renorm_strategy == 'threshold':
                    preds = model(methylation,
                                apply_renorm=True,
                                renorm_strategy='threshold',
                                renorm_params={'threshold': threshold})
                elif renorm_strategy == 'soft_threshold':
                    preds = model(methylation,
                                apply_renorm=True,
                                renorm_strategy='soft_threshold',
                                renorm_params={'threshold': threshold, 
                                             'temperature': 10.0})
                elif renorm_strategy == 'bayesian':
                    preds = model(methylation,
                                apply_renorm=True,
                                renorm_strategy='bayesian',
                                renorm_params={'prior_sparsity': 0.7})
            else:
                preds = model(methylation)
            
            all_preds.append(preds.cpu())
            all_true.append(true_props.cpu())
    
    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_true = torch.cat(all_true, dim=0)
    
    # Compute metrics
    mae = (all_preds - all_true).abs().mean().item()
    
    # Per-tissue MAE
    per_tissue_mae = (all_preds - all_true).abs().mean(dim=0)
    
    # Correlation
    if len(all_preds) > 1:
        correlation = torch.corrcoef(
            torch.stack([all_preds.flatten(), all_true.flatten()])
        )[0, 1].item()
    else:
        correlation = 0.0
    
    metrics = {
        'mae': mae,
        'per_tissue_mae': per_tissue_mae.numpy(),
        'correlation': correlation,
        'n_samples': len(all_preds)
    }
    
    return metrics


# Add to your evaluation loop after training
print("\n" + "="*80)
print("Comparing Renormalization Strategies")
print("="*80)

# Baseline (no renorm)
metrics_baseline = evaluate_with_renorm(
    model, val_loader, device, 
    apply_renorm=False
)
print(f"\nBaseline (no renorm):")
print(f"  MAE: {metrics_baseline['mae']:.6f}")
print(f"  Correlation: {metrics_baseline['correlation']:.4f}")

# Test different thresholds
for thresh in [0.03, 0.05, 0.10]:
    metrics = evaluate_with_renorm(
        model, val_loader, device,
        apply_renorm=True,
        renorm_strategy='threshold',
        threshold=thresh
    )
    improvement = (metrics_baseline['mae'] - metrics['mae']) / metrics_baseline['mae'] * 100
    print(f"\nThreshold {thresh*100:.0f}%:")
    print(f"  MAE: {metrics['mae']:.6f} ({improvement:+.1f}% improvement)")
    print(f"  Correlation: {metrics['correlation']:.4f}")
```

---

## Working with Numpy Arrays

All functions work with both PyTorch tensors and NumPy arrays:

```python
import numpy as np
from model_deconvolution_updated import threshold_renormalize

# NumPy array input
predictions_np = np.array([0.45, 0.02, 0.48, 0.03, 0.02])
renormalized_np = threshold_renormalize(predictions_np, threshold=0.05)
# Returns NumPy array

# PyTorch tensor input
predictions_torch = torch.tensor([0.45, 0.02, 0.48, 0.03, 0.02])
renormalized_torch = threshold_renormalize(predictions_torch, threshold=0.05)
# Returns PyTorch tensor
```

---

## Single Sample vs Batch

Functions handle both single samples and batches:

```python
# Single sample [22]
single_pred = model(single_data)[0]  # [22]
renorm_single = threshold_renormalize(single_pred, threshold=0.05)  # [22]

# Batch [batch_size, 22]
batch_preds = model(batch_data)  # [4, 22]
renorm_batch = threshold_renormalize(batch_preds, threshold=0.05)  # [4, 22]
```

---

## Common Parameter Values

### Threshold (for hard and soft threshold)
- **1%** (0.01): Very permissive, keeps almost everything
- **3%** (0.03): Moderate, good for Phase 2 with 3-5 tissues
- **5%** (0.05): **RECOMMENDED START** - good balance
- **10%** (0.10): Aggressive, only keeps strong predictions

### Temperature (for soft threshold only)
- **5.0**: Gradual transition (softest)
- **10.0**: **RECOMMENDED** - balanced
- **20.0**: Sharp transition (closer to hard threshold)

### Prior Sparsity (for Bayesian)
- **0.5** (50% absent): Low sparsity, more permissive
- **0.7** (70% absent): **RECOMMENDED for Phase 2** - typical mixture
- **0.8** (80% absent): High sparsity, more aggressive

---

## Debugging

### Check proportion sums

```python
# Before renormalization
print(f"Sum before: {predictions.sum(dim=1)}")  # Should be 1.0

# After renormalization
renorm = threshold_renormalize(predictions, threshold=0.05)
print(f"Sum after: {renorm.sum(dim=1)}")  # Should still be 1.0
```

### Count non-zero predictions

```python
# Number of tissues with predictions > 0
n_nonzero = (renorm > 0).sum(dim=1)
print(f"Non-zero tissues: {n_nonzero}")

# Number of tissues with predictions > 5%
n_above_5 = (renorm > 0.05).sum(dim=1)
print(f"Tissues >5%: {n_above_5}")
```

### Visualize effect

```python
import matplotlib.pyplot as plt

# Compare before and after
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Before
axes[0].bar(range(22), raw_predictions[0].cpu().numpy())
axes[0].set_title('Before Renormalization')
axes[0].set_xlabel('Tissue Index')
axes[0].set_ylabel('Proportion')
axes[0].axhline(y=0.05, color='r', linestyle='--', label='Threshold')
axes[0].legend()

# After
axes[1].bar(range(22), renorm_predictions[0].cpu().numpy())
axes[1].set_title('After Renormalization (5%)')
axes[1].set_xlabel('Tissue Index')
axes[1].set_ylabel('Proportion')

plt.tight_layout()
plt.savefig('renormalization_effect.png')
```

---

## FAQ

**Q: Do I need to retrain the model?**  
A: No! These are post-processing steps applied after prediction.

**Q: Which strategy should I use?**  
A: Start with hard threshold at 5%. If not satisfied, try the comparison function.

**Q: Can I use different thresholds for different samples?**  
A: Yes, but you'd need to implement adaptive thresholding. The Bayesian approach effectively does this.

**Q: Will this hurt performance on minor components?**  
A: Possibly. That's why we recommend starting at 5%. Test on validation set.

**Q: Can I apply this during training?**  
A: Not recommended. These are for inference/evaluation only. For training-time solutions, see the usage guide.

**Q: How do I save the best parameters?**  
A: Add to your config file after finding optimal values on validation set.

---

## Summary

**Recommended workflow:**
1. ✅ Start: Hard threshold at 5%
2. ✅ If unsatisfactory: Try [3%, 5%, 10%]
3. ✅ Still unsatisfactory: Compare all three strategies
4. ✅ Use best on test set

**Most common solution:**
```python
predictions = model(data, 
                   apply_renorm=True,
                   renorm_strategy='threshold',
                   renorm_params={'threshold': 0.05})
```

That's it! 🎉
