#!/usr/bin/env python3
"""
Stage 2 Evaluation - With Rescaling and Full Visualization
===========================================================
Handles model trained with normalized outputs (sum=1.0) but test data with absolute labels (sum<1.0)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import h5py
import json
import argparse
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import sys

sys.path.append('.')

# Import model
try:
    from model_deconvolution_absolute import TissueBERTDeconvolution
    HAS_ABSOLUTE = True
except ImportError:
    from model_deconvolution import TissueBERTDeconvolution
    HAS_ABSOLUTE = False

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_model(checkpoint_path, device='cpu', normalize_output=True):
    """Load model"""
    if HAS_ABSOLUTE:
        model = TissueBERTDeconvolution(num_classes=21, n_regions=51089, normalize_output=normalize_output).to(device)
    else:
        model = TissueBERTDeconvolution(num_classes=21, n_regions=51089).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def load_data(test_h5):
    """Load test data"""
    with h5py.File(test_h5, 'r') as f:
        mixed_meth = f['mixed_methylation'][:]
        true_props = f['true_proportions'][:]
        tissue_names = [t.decode() if isinstance(t, bytes) else t for t in f.attrs['tissue_names']]
    return mixed_meth, true_props, tissue_names


def run_inference(model, mixed_meth, device='cpu', batch_size=32):
    """Run inference"""
    all_preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(mixed_meth), batch_size):
            batch_end = min(i + batch_size, len(mixed_meth))
            batch = np.repeat(mixed_meth[i:batch_end, :, np.newaxis], 150, axis=2)
            preds = model(torch.tensor(batch, dtype=torch.float32).to(device))
            all_preds.append(preds.cpu().numpy())
    return np.vstack(all_preds)


def compute_metrics_with_rescaling(predictions, true_proportions, tissue_names, rescale_method='predictions'):
    """Compute metrics with rescaling"""
    pred_sums = predictions.sum(axis=1)
    true_sums = true_proportions.sum(axis=1)
    
    print(f"\n{'='*80}")
    print("SCALE ANALYSIS")
    print(f"{'='*80}")
    print(f"Prediction scale: {pred_sums.mean():.4f} [{pred_sums.min():.4f}, {pred_sums.max():.4f}]")
    print(f"True label scale: {true_sums.mean():.4f} [{true_sums.min():.4f}, {true_sums.max():.4f}]")
    print(f"Scale ratio (pred/true): {pred_sums.mean() / true_sums.mean():.4f}")
    
    # Rescale
    if rescale_method == 'predictions':
        scale_factor = true_sums.mean() / pred_sums.mean()
        predictions_rescaled = predictions * scale_factor
        true_rescaled = true_proportions
        print(f"Scaling predictions by {scale_factor:.4f}")
    elif rescale_method == 'labels':
        true_rescaled = true_proportions / true_proportions.sum(axis=1, keepdims=True)
        predictions_rescaled = predictions
        print(f"Normalizing labels to sum=1.0")
    else:
        predictions_rescaled = predictions
        true_rescaled = true_proportions
    
    # Overall metrics
    mae = mean_absolute_error(true_rescaled.flatten(), predictions_rescaled.flatten())
    rmse = np.sqrt(mean_squared_error(true_rescaled.flatten(), predictions_rescaled.flatten()))
    r2 = r2_score(true_rescaled.flatten(), predictions_rescaled.flatten())
    pearson_r, _ = pearsonr(true_rescaled.flatten(), predictions_rescaled.flatten())
    spearman_r, _ = spearmanr(true_rescaled.flatten(), predictions_rescaled.flatten())
    
    print(f"\n{'='*80}")
    print("OVERALL METRICS")
    print(f"{'='*80}")
    print(f"MAE:  {mae:.6f} ({mae*100:.2f}%)")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²:   {r2:.4f}")
    print(f"Pearson r:  {pearson_r:.4f}")
    print(f"Spearman r: {spearman_r:.4f}")
    
    overall_metrics = {'mae': mae, 'rmse': rmse, 'r2': r2, 'pearson_r': pearson_r, 'spearman_r': spearman_r}
    
    # Per-tissue metrics
    per_tissue_results = []
    for i, tissue in enumerate(tissue_names):
        true_vals = true_rescaled[:, i]
        pred_vals = predictions_rescaled[:, i]
        present_mask = true_vals > 0
        n_present = present_mask.sum()
        
        if n_present > 0:
            mae_present = mean_absolute_error(true_vals[present_mask], pred_vals[present_mask])
            r2_present = r2_score(true_vals[present_mask], pred_vals[present_mask]) if n_present > 10 else np.nan
        else:
            mae_present = np.nan
            r2_present = np.nan
        
        per_tissue_results.append({
            'tissue': tissue,
            'mae_present': mae_present,
            'r2_present': r2_present,
            'n_present': n_present,
            'frequency': n_present / len(true_vals)
        })
    
    df_per_tissue = pd.DataFrame(per_tissue_results)
    return overall_metrics, df_per_tissue, predictions_rescaled, true_rescaled


def save_results(output_dir, metrics, df_per_tissue, predictions, true_proportions, tissue_names):
    """Save results - FIXED VERSION with proper JSON serialization"""
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    
    # Convert numpy types to Python native types for JSON serialization
    metrics_serializable = {}
    for k, v in metrics.items():
        if isinstance(v, (np.integer, np.floating, np.int32, np.int64, np.float32, np.float64)):
            metrics_serializable[k] = float(v)
        elif isinstance(v, np.ndarray):
            metrics_serializable[k] = float(v.item()) if v.size == 1 else v.tolist()
        else:
            metrics_serializable[k] = v
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"  ✓ metrics.json")
    
    df_per_tissue.to_csv(output_dir / 'per_tissue_metrics.csv', index=False)
    print(f"  ✓ per_tissue_metrics.csv")
    
    np.save(output_dir / 'predictions.npy', predictions)
    np.save(output_dir / 'true_proportions.npy', true_proportions)
    print(f"  ✓ predictions.npy, true_proportions.npy")
    
    with open(output_dir / 'tissue_names.txt', 'w') as f:
        f.write('\n'.join(tissue_names))
    print(f"  ✓ tissue_names.txt")


def plot_overall_performance(predictions, true_proportions, metrics, figures_dir):
    """Plot overall performance"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ax = axes[0]
    ax.scatter(true_proportions.flatten(), predictions.flatten(), alpha=0.3, s=1)
    ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect')
    ax.set_xlabel('True'); ax.set_ylabel('Predicted')
    ax.set_title(f'Predictions vs Ground Truth\nMAE={metrics["mae"]:.4f}, R²={metrics["r2"]:.3f}')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    residuals = predictions.flatten() - true_proportions.flatten()
    ax.hist(residuals, bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Residual'); ax.set_ylabel('Frequency')
    ax.set_title(f'Residuals (Mean={residuals.mean():.4f})')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    per_sample_mae = np.abs(predictions - true_proportions).mean(axis=1)
    ax.hist(per_sample_mae, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(per_sample_mae.mean(), color='r', linestyle='--', lw=2, label=f'Mean={per_sample_mae.mean():.4f}')
    ax.set_xlabel('Per-Sample MAE'); ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution'); ax.legend(); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'overall_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ overall_performance.png")


def plot_per_tissue_performance(predictions, true_proportions, tissue_names, df_per_tissue, figures_dir):
    """Plot per-tissue performance"""
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    axes = axes.flatten()
    
    for i, tissue in enumerate(tissue_names):
        if i >= len(axes): break
        ax = axes[i]
        true_vals = true_proportions[:, i]
        pred_vals = predictions[:, i]
        present_mask = true_vals > 0
        
        if present_mask.sum() > 0:
            ax.scatter(true_vals[present_mask], pred_vals[present_mask], alpha=0.5, s=10)
            ax.plot([0, true_vals.max()], [0, true_vals.max()], 'r--', lw=1)
            metrics = df_per_tissue[df_per_tissue['tissue'] == tissue].iloc[0]
            ax.set_title(f'{tissue}\nMAE={metrics["mae_present"]:.3f}, R²={metrics["r2_present"]:.2f}', fontsize=9)
            ax.set_xlabel('True'); ax.set_ylabel('Predicted')
            ax.grid(True, alpha=0.3)
    
    for i in range(len(tissue_names), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'per_tissue_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ per_tissue_performance.png")


def main():
    parser = argparse.ArgumentParser(description='Stage 2 Evaluation with Rescaling')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--test_h5', required=True, help='Test HDF5 path')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--device', default='cpu', choices=['cuda', 'cpu'])
    parser.add_argument('--rescale', default='predictions', choices=['predictions', 'labels', 'none'])
    args = parser.parse_args()
    
    # Create directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("STAGE 2 EVALUATION - WITH RESCALING")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test data: {args.test_h5}")
    print(f"Output: {args.output_dir}")
    print(f"Figures: {figures_dir}")
    
    # Load and run
    print("\nLoading model...")
    model, checkpoint = load_model(args.checkpoint, args.device, normalize_output=True)
    print(f"  ✓ Epoch {checkpoint.get('epoch')}, Val MAE {checkpoint.get('val_mae', 0):.4f}")
    
    print("\nLoading test data...")
    mixed_meth, true_props, tissue_names = load_data(args.test_h5)
    print(f"  ✓ {len(mixed_meth)} samples, {len(tissue_names)} tissues")
    
    print("\nRunning inference...")
    predictions = run_inference(model, mixed_meth, args.device)
    print(f"  ✓ Predictions shape: {predictions.shape}")
    
    print("\nComputing metrics...")
    metrics, df_per_tissue, pred_rescaled, true_rescaled = compute_metrics_with_rescaling(
        predictions, true_props, tissue_names, args.rescale
    )
    
    # Save results
    save_results(output_dir, metrics, df_per_tissue, pred_rescaled, true_rescaled, tissue_names)
    
    # Generate figures
    print(f"\n{'='*80}")
    print("GENERATING FIGURES")
    print(f"{'='*80}")
    plot_overall_performance(pred_rescaled, true_rescaled, metrics, figures_dir)
    plot_per_tissue_performance(pred_rescaled, true_rescaled, tissue_names, df_per_tissue, figures_dir)
    
    # Summary
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"MAE: {metrics['mae']:.4f} ({metrics['mae']*100:.2f}%)")
    print(f"R²:  {metrics['r2']:.4f}")
    print(f"\nResults: {output_dir}")
    print(f"Figures: {figures_dir}")
    
    if metrics['mae'] < 0.06 and metrics['r2'] > 0.80:
        print("\n✓ EXCELLENT: Matches validation performance!")
    elif metrics['mae'] < 0.08:
        print("\n✓ GOOD: Reasonable performance")
    else:
        print("\n⚠️  Poor performance - check rescaling or model/data mismatch")
    
    print("="*80)


if __name__ == '__main__':
    main()
