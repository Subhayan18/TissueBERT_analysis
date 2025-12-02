#!/usr/bin/env python3
"""
Comprehensive Evaluation and Visualization for Mixture Deconvolution
=====================================================================

Generate extensive reports and detailed comparative figures for trained models.

Features:
- Overall performance metrics (MAE, MSE, correlation)
- Per-tissue analysis with detailed statistics
- Prediction vs ground truth scatter plots
- Error distribution analysis
- Heatmaps (confusion-style for proportions)
- Per-mixture-type analysis (2-tissue, 3-tissue, etc.)
- Top errors and best predictions
- Residual analysis
- Tissue frequency and coverage statistics

Works for all phases (1, 2, 3).

Author: Mixture Deconvolution Project
Date: December 2024
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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import model
import sys
sys.path.append('.')
from model_deconvolution import load_pretrained_model

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_model_and_data(checkpoint_path, test_h5, device='cuda'):
    """Load trained model and test data"""
    print(f"\n{'='*80}")
    print("LOADING MODEL AND DATA")
    print(f"{'='*80}")
    
    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    model = load_pretrained_model(checkpoint_path, device=device, verbose=False)
    model.eval()
    print("✓ Model loaded")
    
    # Load test data
    print(f"\nLoading test data from: {test_h5}")
    with h5py.File(test_h5, 'r') as f:
        mixed_methylation = f['mixed_methylation'][:]
        true_proportions = f['true_proportions'][:]
        tissue_names = [t.decode() if isinstance(t, bytes) else t 
                       for t in f.attrs['tissue_names']]
        phase = f.attrs['phase']
        n_mixtures = f.attrs['n_mixtures']
        
        # Load mixture info if available
        try:
            mixture_info_json = f['mixture_info'][:]
            mixture_info = [json.loads(s) for s in mixture_info_json]
        except:
            mixture_info = None
    
    print(f"✓ Loaded {n_mixtures} test mixtures")
    print(f"  Phase: {phase}")
    print(f"  Tissues: {len(tissue_names)}")
    
    return model, mixed_methylation, true_proportions, tissue_names, phase, mixture_info


def run_inference(model, mixed_methylation, device='cuda', batch_size=32):
    """Run inference on test data"""
    print(f"\n{'='*80}")
    print("RUNNING INFERENCE")
    print(f"{'='*80}")
    
    n_samples = len(mixed_methylation)
    seq_length = 150
    
    all_predictions = []
    
    # Process in batches
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            
            # Expand to [batch, n_regions, seq_length]
            batch_means = mixed_methylation[i:batch_end]
            batch_meth = np.repeat(batch_means[:, :, np.newaxis], seq_length, axis=2)
            
            # Convert to tensor
            batch_tensor = torch.tensor(batch_meth, dtype=torch.float32).to(device)
            
            # Predict
            proportions = model(batch_tensor)
            
            all_predictions.append(proportions.cpu().numpy())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {batch_end}/{n_samples} samples...")
    
    predictions = np.vstack(all_predictions)
    
    print(f"\n✓ Inference complete")
    print(f"  Predictions shape: {predictions.shape}")
    
    return predictions


def compute_overall_metrics(predictions, true_proportions):
    """Compute overall performance metrics"""
    print(f"\n{'='*80}")
    print("COMPUTING OVERALL METRICS")
    print(f"{'='*80}")
    
    metrics = {}
    
    # MSE
    mse = mean_squared_error(true_proportions, predictions)
    metrics['mse'] = float(mse)
    
    # MAE
    mae = mean_absolute_error(true_proportions, predictions)
    metrics['mae'] = float(mae)
    
    # RMSE
    rmse = np.sqrt(mse)
    metrics['rmse'] = float(rmse)
    
    # R²
    r2 = r2_score(true_proportions.flatten(), predictions.flatten())
    metrics['r2'] = float(r2)
    
    # Pearson correlation
    pearson_r, pearson_p = pearsonr(true_proportions.flatten(), predictions.flatten())
    metrics['pearson_r'] = float(pearson_r)
    metrics['pearson_p'] = float(pearson_p)
    
    # Spearman correlation
    spearman_r, spearman_p = spearmanr(true_proportions.flatten(), predictions.flatten())
    metrics['spearman_r'] = float(spearman_r)
    metrics['spearman_p'] = float(spearman_p)
    
    # Per-sample metrics
    per_sample_mae = np.abs(predictions - true_proportions).mean(axis=1)
    metrics['per_sample_mae_mean'] = float(per_sample_mae.mean())
    metrics['per_sample_mae_std'] = float(per_sample_mae.std())
    metrics['per_sample_mae_median'] = float(np.median(per_sample_mae))
    metrics['per_sample_mae_max'] = float(per_sample_mae.max())
    metrics['per_sample_mae_min'] = float(per_sample_mae.min())
    
    # Proportion sum check
    pred_sums = predictions.sum(axis=1)
    metrics['pred_sum_mean'] = float(pred_sums.mean())
    metrics['pred_sum_std'] = float(pred_sums.std())
    metrics['pred_sum_all_valid'] = bool(np.allclose(pred_sums, 1.0, atol=1e-5))
    
    print(f"\nOverall Metrics:")
    print(f"  MAE: {metrics['mae']:.6f} ({metrics['mae']*100:.2f}%)")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Pearson r: {metrics['pearson_r']:.4f}")
    print(f"  Spearman r: {metrics['spearman_r']:.4f}")
    
    return metrics, per_sample_mae


def compute_per_tissue_metrics(predictions, true_proportions, tissue_names):
    """Compute per-tissue performance metrics"""
    print(f"\n{'='*80}")
    print("COMPUTING PER-TISSUE METRICS")
    print(f"{'='*80}")
    
    n_tissues = len(tissue_names)
    per_tissue_metrics = []
    
    for i, tissue in enumerate(tissue_names):
        true_vals = true_proportions[:, i]
        pred_vals = predictions[:, i]
        
        # Only analyze samples where tissue is present
        present_mask = true_vals > 0
        n_present = present_mask.sum()
        
        if n_present > 0:
            # Metrics on present samples
            mae_present = mean_absolute_error(true_vals[present_mask], pred_vals[present_mask])
            rmse_present = np.sqrt(mean_squared_error(true_vals[present_mask], pred_vals[present_mask]))
            r2_present = r2_score(true_vals[present_mask], pred_vals[present_mask]) if n_present > 1 else 0
            
            if n_present > 2:
                pearson_r, pearson_p = pearsonr(true_vals[present_mask], pred_vals[present_mask])
            else:
                pearson_r, pearson_p = 0, 1
        else:
            mae_present = 0
            rmse_present = 0
            r2_present = 0
            pearson_r = 0
            pearson_p = 1
        
        # Overall metrics (all samples)
        mae_all = mean_absolute_error(true_vals, pred_vals)
        rmse_all = np.sqrt(mean_squared_error(true_vals, pred_vals))
        
        # Mean values
        true_mean = true_vals[present_mask].mean() if n_present > 0 else 0
        pred_mean = pred_vals[present_mask].mean() if n_present > 0 else 0
        
        # Bias
        bias = (pred_vals[present_mask] - true_vals[present_mask]).mean() if n_present > 0 else 0
        
        per_tissue_metrics.append({
            'tissue': tissue,
            'n_present': int(n_present),
            'n_total': len(true_vals),
            'frequency': n_present / len(true_vals),
            'mae_present': float(mae_present),
            'rmse_present': float(rmse_present),
            'r2_present': float(r2_present),
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'mae_all': float(mae_all),
            'rmse_all': float(rmse_all),
            'true_mean': float(true_mean),
            'pred_mean': float(pred_mean),
            'bias': float(bias)
        })
    
    df_per_tissue = pd.DataFrame(per_tissue_metrics)
    
    print(f"\nTop 5 tissues by MAE (when present):")
    top5 = df_per_tissue.nlargest(5, 'mae_present')[['tissue', 'mae_present', 'n_present']]
    for _, row in top5.iterrows():
        print(f"  {row['tissue']}: MAE={row['mae_present']:.4f} (n={row['n_present']})")
    
    print(f"\nBottom 5 tissues by MAE (when present):")
    bottom5 = df_per_tissue.nsmallest(5, 'mae_present')[['tissue', 'mae_present', 'n_present']]
    for _, row in bottom5.iterrows():
        print(f"  {row['tissue']}: MAE={row['mae_present']:.4f} (n={row['n_present']})")
    
    return df_per_tissue


def analyze_by_mixture_type(predictions, true_proportions, mixture_info):
    """Analyze performance by mixture type (2-tissue, 3-tissue, etc.)"""
    if mixture_info is None:
        return None
    
    print(f"\n{'='*80}")
    print("ANALYZING BY MIXTURE TYPE")
    print(f"{'='*80}")
    
    # Determine number of tissues per mixture
    n_tissues_per_mixture = (true_proportions > 0).sum(axis=1)
    
    mixture_type_metrics = []
    
    for n_tissues in sorted(np.unique(n_tissues_per_mixture)):
        mask = n_tissues_per_mixture == n_tissues
        n_samples = mask.sum()
        
        if n_samples > 0:
            mae = mean_absolute_error(
                true_proportions[mask].flatten(),
                predictions[mask].flatten()
            )
            
            rmse = np.sqrt(mean_squared_error(
                true_proportions[mask].flatten(),
                predictions[mask].flatten()
            ))
            
            r2 = r2_score(
                true_proportions[mask].flatten(),
                predictions[mask].flatten()
            )
            
            mixture_type_metrics.append({
                'n_tissues': int(n_tissues),
                'n_samples': int(n_samples),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2)
            })
    
    df_mixture_type = pd.DataFrame(mixture_type_metrics)
    
    print(f"\nPerformance by mixture complexity:")
    for _, row in df_mixture_type.iterrows():
        print(f"  {row['n_tissues']} tissues: MAE={row['mae']:.4f}, n={row['n_samples']}")
    
    return df_mixture_type


def find_best_worst_predictions(predictions, true_proportions, n_samples=10):
    """Find best and worst predictions"""
    per_sample_mae = np.abs(predictions - true_proportions).mean(axis=1)
    
    # Best predictions (lowest MAE)
    best_indices = np.argsort(per_sample_mae)[:n_samples]
    best_mae = per_sample_mae[best_indices]
    
    # Worst predictions (highest MAE)
    worst_indices = np.argsort(per_sample_mae)[-n_samples:][::-1]
    worst_mae = per_sample_mae[worst_indices]
    
    return best_indices, best_mae, worst_indices, worst_mae


def plot_overall_performance(predictions, true_proportions, metrics, output_dir):
    """Plot overall performance metrics"""
    print(f"\n{'='*80}")
    print("GENERATING OVERALL PERFORMANCE PLOTS")
    print(f"{'='*80}")
    
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True, parents=True)
    
    # Figure 1: Predicted vs True (all data points)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot
    ax = axes[0]
    ax.scatter(true_proportions.flatten(), predictions.flatten(), 
              alpha=0.3, s=10, edgecolors='none')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('True Proportion', fontsize=12)
    ax.set_ylabel('Predicted Proportion', fontsize=12)
    ax.set_title(f'Predicted vs True Proportions\nMAE={metrics["mae"]:.4f}, R²={metrics["r2"]:.3f}', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    
    # Hexbin plot (density)
    ax = axes[1]
    hb = ax.hexbin(true_proportions.flatten(), predictions.flatten(), 
                   gridsize=50, cmap='Blues', mincnt=1)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('True Proportion', fontsize=12)
    ax.set_ylabel('Predicted Proportion', fontsize=12)
    ax.set_title('Density Plot: Predicted vs True', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    plt.colorbar(hb, ax=ax, label='Count')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig1_overall_performance.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: fig1_overall_performance.png")
    
    # Figure 2: Error distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Histogram of errors
    ax = axes[0, 0]
    errors = (predictions - true_proportions).flatten()
    ax.hist(errors, bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax.set_xlabel('Prediction Error (Pred - True)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Error Distribution\nMean={errors.mean():.4f}, Std={errors.std():.4f}', 
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Absolute errors
    ax = axes[0, 1]
    abs_errors = np.abs(errors)
    ax.hist(abs_errors, bins=100, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(x=metrics['mae'], color='red', linestyle='--', linewidth=2, 
              label=f'MAE={metrics["mae"]:.4f}')
    ax.set_xlabel('Absolute Error |Pred - True|', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Absolute Error Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Per-sample MAE
    ax = axes[1, 0]
    per_sample_mae = np.abs(predictions - true_proportions).mean(axis=1)
    ax.hist(per_sample_mae, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(x=per_sample_mae.mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean={per_sample_mae.mean():.4f}')
    ax.set_xlabel('Per-Sample MAE', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Per-Sample MAE', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Residuals vs predicted
    ax = axes[1, 1]
    ax.scatter(predictions.flatten(), errors, alpha=0.3, s=10, edgecolors='none')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Proportion', fontsize=12)
    ax.set_ylabel('Residual (Pred - True)', fontsize=12)
    ax.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig2_error_analysis.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: fig2_error_analysis.png")


def plot_per_tissue_performance(predictions, true_proportions, tissue_names, 
                                df_per_tissue, output_dir):
    """Plot per-tissue performance"""
    print(f"\n{'='*80}")
    print("GENERATING PER-TISSUE PERFORMANCE PLOTS")
    print(f"{'='*80}")
    
    fig_dir = output_dir / 'figures'
    n_tissues = len(tissue_names)
    
    # Figure 3: Per-tissue scatter plots (grid)
    n_cols = 5
    n_rows = int(np.ceil(n_tissues / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten()
    
    for i, tissue in enumerate(tissue_names):
        ax = axes[i]
        
        true_vals = true_proportions[:, i]
        pred_vals = predictions[:, i]
        
        # Only plot samples where tissue is present
        present_mask = true_vals > 0
        
        if present_mask.sum() > 0:
            ax.scatter(true_vals[present_mask], pred_vals[present_mask], 
                      alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
            ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5)
            
            # Get metrics
            row = df_per_tissue[df_per_tissue['tissue'] == tissue].iloc[0]
            mae = row['mae_present']
            r2 = row['r2_present']
            n = row['n_present']
            
            ax.set_xlabel('True Proportion', fontsize=9)
            ax.set_ylabel('Predicted Proportion', fontsize=9)
            ax.set_title(f'{tissue}\nMAE={mae:.3f}, R²={r2:.2f}, n={n}', 
                        fontsize=9, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([-0.05, 1.05])
            ax.set_ylim([-0.05, 1.05])
        else:
            ax.text(0.5, 0.5, f'{tissue}\nNo samples', 
                   ha='center', va='center', fontsize=10)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
    
    # Hide extra subplots
    for i in range(n_tissues, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig3_per_tissue_scatter.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: fig3_per_tissue_scatter.png")
    
    # Figure 4: Per-tissue MAE bar chart
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Sort by MAE (present)
    df_sorted = df_per_tissue.sort_values('mae_present', ascending=False)
    
    colors = ['red' if mae > 0.1 else 'orange' if mae > 0.05 else 'green' 
             for mae in df_sorted['mae_present']]
    
    bars = ax.barh(range(len(df_sorted)), df_sorted['mae_present'], 
                   color=colors, edgecolor='black', alpha=0.7)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['tissue'], fontsize=10)
    ax.set_xlabel('MAE (when present)', fontsize=12)
    ax.set_title('Per-Tissue MAE (sorted)', fontsize=14, fontweight='bold')
    ax.axvline(x=0.05, color='blue', linestyle='--', linewidth=2, 
              label='5% threshold')
    ax.axvline(x=0.1, color='red', linestyle='--', linewidth=2,
              label='10% threshold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Annotate with sample counts
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        ax.text(row['mae_present'] + 0.005, i, f"n={row['n_present']}", 
               va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig4_per_tissue_mae.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: fig4_per_tissue_mae.png")
    
    # Figure 5: Tissue frequency and performance
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Frequency
    ax = axes[0]
    df_sorted_freq = df_per_tissue.sort_values('frequency', ascending=False)
    bars = ax.bar(range(len(df_sorted_freq)), df_sorted_freq['frequency'],
                 edgecolor='black', alpha=0.7)
    ax.set_xticks(range(len(df_sorted_freq)))
    ax.set_xticklabels(df_sorted_freq['tissue'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Frequency (proportion of samples)', fontsize=12)
    ax.set_title('Tissue Frequency in Test Set', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # MAE vs Frequency
    ax = axes[1]
    ax.scatter(df_per_tissue['frequency'], df_per_tissue['mae_present'],
              s=100, alpha=0.6, edgecolors='black')
    
    for _, row in df_per_tissue.iterrows():
        ax.annotate(row['tissue'], 
                   (row['frequency'], row['mae_present']),
                   fontsize=7, ha='center')
    
    ax.set_xlabel('Tissue Frequency', fontsize=12)
    ax.set_ylabel('MAE (when present)', fontsize=12)
    ax.set_title('MAE vs Tissue Frequency', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig5_tissue_frequency.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: fig5_tissue_frequency.png")


def plot_heatmaps(predictions, true_proportions, tissue_names, output_dir):
    """Plot heatmaps for true and predicted proportions"""
    print(f"\n{'='*80}")
    print("GENERATING HEATMAPS")
    print(f"{'='*80}")
    
    fig_dir = output_dir / 'figures'
    
    # Limit to first 100 samples for readability
    n_samples_plot = min(100, len(predictions))
    
    # Figure 6: Side-by-side heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    
    # True proportions
    ax = axes[0]
    im = ax.imshow(true_proportions[:n_samples_plot].T, aspect='auto', 
                   cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Tissue', fontsize=12)
    ax.set_title(f'True Proportions (first {n_samples_plot} samples)', 
                fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(tissue_names)))
    ax.set_yticklabels(tissue_names, fontsize=9)
    plt.colorbar(im, ax=ax, label='Proportion')
    
    # Predicted proportions
    ax = axes[1]
    im = ax.imshow(predictions[:n_samples_plot].T, aspect='auto',
                   cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Tissue', fontsize=12)
    ax.set_title(f'Predicted Proportions (first {n_samples_plot} samples)',
                fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(tissue_names)))
    ax.set_yticklabels(tissue_names, fontsize=9)
    plt.colorbar(im, ax=ax, label='Proportion')
    
    # Difference (Predicted - True)
    ax = axes[2]
    diff = predictions[:n_samples_plot] - true_proportions[:n_samples_plot]
    im = ax.imshow(diff.T, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Tissue', fontsize=12)
    ax.set_title(f'Error (Pred - True)', fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(tissue_names)))
    ax.set_yticklabels(tissue_names, fontsize=9)
    plt.colorbar(im, ax=ax, label='Error')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig6_heatmaps.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: fig6_heatmaps.png")
    
    # Figure 7: Correlation heatmap (tissue-tissue)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # True correlations
    ax = axes[0]
    true_corr = np.corrcoef(true_proportions.T)
    im = ax.imshow(true_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(tissue_names)))
    ax.set_xticklabels(tissue_names, rotation=90, fontsize=8)
    ax.set_yticks(range(len(tissue_names)))
    ax.set_yticklabels(tissue_names, fontsize=8)
    ax.set_title('True Proportion Correlations\n(Tissue Co-occurrence)', 
                fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Correlation')
    
    # Predicted correlations
    ax = axes[1]
    pred_corr = np.corrcoef(predictions.T)
    im = ax.imshow(pred_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(tissue_names)))
    ax.set_xticklabels(tissue_names, rotation=90, fontsize=8)
    ax.set_yticks(range(len(tissue_names)))
    ax.set_yticklabels(tissue_names, fontsize=8)
    ax.set_title('Predicted Proportion Correlations',
                fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Correlation')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig7_correlation_heatmap.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: fig7_correlation_heatmap.png")


def plot_best_worst_predictions(predictions, true_proportions, tissue_names,
                                best_indices, best_mae, worst_indices, worst_mae,
                                output_dir):
    """Plot best and worst predictions"""
    print(f"\n{'='*80}")
    print("GENERATING BEST/WORST PREDICTION PLOTS")
    print(f"{'='*80}")
    
    fig_dir = output_dir / 'figures'
    
    n_show = min(5, len(best_indices))
    
    # Figure 8: Best predictions
    fig, axes = plt.subplots(1, n_show, figsize=(4*n_show, 5))
    if n_show == 1:
        axes = [axes]
    
    for i in range(n_show):
        idx = best_indices[i]
        ax = axes[i]
        
        x = np.arange(len(tissue_names))
        width = 0.35
        
        ax.bar(x - width/2, true_proportions[idx], width, 
              label='True', alpha=0.7, edgecolor='black')
        ax.bar(x + width/2, predictions[idx], width,
              label='Predicted', alpha=0.7, edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels(tissue_names, rotation=90, fontsize=8)
        ax.set_ylabel('Proportion', fontsize=10)
        ax.set_title(f'Best #{i+1}\nMAE={best_mae[i]:.4f}', 
                    fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig8_best_predictions.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: fig8_best_predictions.png")
    
    # Figure 9: Worst predictions
    fig, axes = plt.subplots(1, n_show, figsize=(4*n_show, 5))
    if n_show == 1:
        axes = [axes]
    
    for i in range(n_show):
        idx = worst_indices[i]
        ax = axes[i]
        
        x = np.arange(len(tissue_names))
        width = 0.35
        
        ax.bar(x - width/2, true_proportions[idx], width,
              label='True', alpha=0.7, edgecolor='black')
        ax.bar(x + width/2, predictions[idx], width,
              label='Predicted', alpha=0.7, edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels(tissue_names, rotation=90, fontsize=8)
        ax.set_ylabel('Proportion', fontsize=10)
        ax.set_title(f'Worst #{i+1}\nMAE={worst_mae[i]:.4f}',
                    fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig9_worst_predictions.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: fig9_worst_predictions.png")


def plot_mixture_type_analysis(predictions, true_proportions, df_mixture_type, output_dir):
    """Plot analysis by mixture type"""
    if df_mixture_type is None or len(df_mixture_type) == 0:
        return
    
    print(f"\n{'='*80}")
    print("GENERATING MIXTURE TYPE ANALYSIS PLOTS")
    print(f"{'='*80}")
    
    fig_dir = output_dir / 'figures'
    
    # Figure 10: Performance by mixture complexity
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # MAE vs complexity
    ax = axes[0]
    ax.bar(df_mixture_type['n_tissues'], df_mixture_type['mae'],
          edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Tissues in Mixture', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('MAE vs Mixture Complexity', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotate with sample counts
    for _, row in df_mixture_type.iterrows():
        ax.text(row['n_tissues'], row['mae'] + 0.005, f"n={row['n_samples']}",
               ha='center', fontsize=9)
    
    # RMSE vs complexity
    ax = axes[1]
    ax.bar(df_mixture_type['n_tissues'], df_mixture_type['rmse'],
          edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel('Number of Tissues in Mixture', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('RMSE vs Mixture Complexity', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # R² vs complexity
    ax = axes[2]
    ax.bar(df_mixture_type['n_tissues'], df_mixture_type['r2'],
          edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel('Number of Tissues in Mixture', fontsize=12)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('R² vs Mixture Complexity', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig10_mixture_complexity.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: fig10_mixture_complexity.png")


def generate_text_report(metrics, df_per_tissue, df_mixture_type, 
                        best_indices, worst_indices, output_dir, phase):
    """Generate comprehensive text report"""
    print(f"\n{'='*80}")
    print("GENERATING TEXT REPORT")
    print(f"{'='*80}")
    
    report = []
    report.append("="*80)
    report.append(f"MIXTURE DECONVOLUTION EVALUATION REPORT - PHASE {phase}")
    report.append("="*80)
    report.append("")
    
    # Overall metrics
    report.append("1. OVERALL PERFORMANCE")
    report.append("-"*80)
    report.append(f"Mean Absolute Error (MAE): {metrics['mae']:.6f} ({metrics['mae']*100:.2f}%)")
    report.append(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.6f}")
    report.append(f"R² Score: {metrics['r2']:.4f}")
    report.append(f"Pearson Correlation: {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.2e})")
    report.append(f"Spearman Correlation: {metrics['spearman_r']:.4f} (p={metrics['spearman_p']:.2e})")
    report.append("")
    report.append(f"Per-sample MAE statistics:")
    report.append(f"  Mean: {metrics['per_sample_mae_mean']:.6f}")
    report.append(f"  Std: {metrics['per_sample_mae_std']:.6f}")
    report.append(f"  Median: {metrics['per_sample_mae_median']:.6f}")
    report.append(f"  Range: [{metrics['per_sample_mae_min']:.6f}, {metrics['per_sample_mae_max']:.6f}]")
    report.append("")
    report.append(f"Prediction sum validation:")
    report.append(f"  Mean sum: {metrics['pred_sum_mean']:.6f}")
    report.append(f"  Std sum: {metrics['pred_sum_std']:.6f}")
    report.append(f"  All valid (sum≈1): {metrics['pred_sum_all_valid']}")
    report.append("")
    
    # Per-tissue performance
    report.append("2. PER-TISSUE PERFORMANCE")
    report.append("-"*80)
    report.append(f"{'Tissue':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'N':<10} {'Freq':<10}")
    report.append("-"*80)
    
    for _, row in df_per_tissue.sort_values('mae_present').iterrows():
        report.append(f"{row['tissue']:<20} "
                     f"{row['mae_present']:<10.4f} "
                     f"{row['rmse_present']:<10.4f} "
                     f"{row['r2_present']:<10.3f} "
                     f"{row['n_present']:<10} "
                     f"{row['frequency']:<10.3f}")
    report.append("")
    
    # Summary statistics
    report.append("Per-tissue summary:")
    report.append(f"  Best tissue (lowest MAE): {df_per_tissue.nsmallest(1, 'mae_present')['tissue'].iloc[0]} "
                 f"(MAE={df_per_tissue['mae_present'].min():.4f})")
    report.append(f"  Worst tissue (highest MAE): {df_per_tissue.nlargest(1, 'mae_present')['tissue'].iloc[0]} "
                 f"(MAE={df_per_tissue['mae_present'].max():.4f})")
    report.append(f"  Mean MAE across tissues: {df_per_tissue['mae_present'].mean():.4f}")
    report.append(f"  Median MAE across tissues: {df_per_tissue['mae_present'].median():.4f}")
    report.append("")
    
    # Mixture type analysis
    if df_mixture_type is not None and len(df_mixture_type) > 0:
        report.append("3. MIXTURE TYPE ANALYSIS")
        report.append("-"*80)
        report.append(f"{'N Tissues':<15} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'N Samples':<15}")
        report.append("-"*80)
        
        for _, row in df_mixture_type.iterrows():
            report.append(f"{row['n_tissues']:<15} "
                         f"{row['mae']:<10.4f} "
                         f"{row['rmse']:<10.4f} "
                         f"{row['r2']:<10.3f} "
                         f"{row['n_samples']:<15}")
        report.append("")
    
    # Best/worst predictions
    report.append("4. BEST AND WORST PREDICTIONS")
    report.append("-"*80)
    report.append(f"Best prediction indices (top 10): {best_indices[:10].tolist()}")
    report.append(f"Worst prediction indices (top 10): {worst_indices[:10].tolist()}")
    report.append("")
    
    # Success criteria
    report.append("5. SUCCESS CRITERIA")
    report.append("-"*80)
    
    if phase == 1:
        target_mae = 0.05
        report.append(f"Phase 1 target: MAE < 5%")
    elif phase == 2:
        target_mae = 0.08
        report.append(f"Phase 2 target: MAE < 8%")
    elif phase == 3:
        target_mae = 0.10
        report.append(f"Phase 3 target: MAE < 10%")
    else:
        target_mae = 0.10
        report.append(f"General target: MAE < 10%")
    
    if metrics['mae'] < target_mae:
        report.append(f"✓ SUCCESS: MAE={metrics['mae']:.4f} < {target_mae}")
    else:
        report.append(f"✗ NEEDS IMPROVEMENT: MAE={metrics['mae']:.4f} >= {target_mae}")
    
    report.append("")
    report.append("="*80)
    
    # Save report
    report_path = output_dir / 'evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\n✓ Saved text report: {report_path}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    for line in report[:20]:  # Print first 20 lines
        print(line)
    print("...")
    print(f"\nFull report saved to: {report_path}")


def save_results(metrics, df_per_tissue, df_mixture_type, predictions, 
                true_proportions, output_dir):
    """Save all results to files"""
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    
    # Save metrics as JSON
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics: {metrics_path}")
    
    # Save per-tissue metrics as CSV
    df_per_tissue.to_csv(output_dir / 'per_tissue_metrics.csv', index=False)
    print(f"✓ Saved per-tissue metrics: per_tissue_metrics.csv")
    
    # Save mixture type metrics if available
    if df_mixture_type is not None:
        df_mixture_type.to_csv(output_dir / 'mixture_type_metrics.csv', index=False)
        print(f"✓ Saved mixture type metrics: mixture_type_metrics.csv")
    
    # Save predictions and true proportions
    np.save(output_dir / 'predictions.npy', predictions)
    np.save(output_dir / 'true_proportions.npy', true_proportions)
    print(f"✓ Saved predictions and ground truth arrays")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive evaluation of mixture deconvolution model'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test_h5', type=str, required=True,
                       help='Path to test mixture HDF5 file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for inference')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("COMPREHENSIVE MIXTURE DECONVOLUTION EVALUATION")
    print("="*80)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Test data: {args.test_h5}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    
    # Load model and data
    model, mixed_methylation, true_proportions, tissue_names, phase, mixture_info = \
        load_model_and_data(args.checkpoint, args.test_h5, args.device)
    
    # Run inference
    predictions = run_inference(model, mixed_methylation, args.device, args.batch_size)
    
    # Compute metrics
    metrics, per_sample_mae = compute_overall_metrics(predictions, true_proportions)
    df_per_tissue = compute_per_tissue_metrics(predictions, true_proportions, tissue_names)
    df_mixture_type = analyze_by_mixture_type(predictions, true_proportions, mixture_info)
    
    # Find best/worst predictions
    best_indices, best_mae, worst_indices, worst_mae = \
        find_best_worst_predictions(predictions, true_proportions, n_samples=10)
    
    # Generate plots
    plot_overall_performance(predictions, true_proportions, metrics, output_dir)
    plot_per_tissue_performance(predictions, true_proportions, tissue_names,
                                df_per_tissue, output_dir)
    plot_heatmaps(predictions, true_proportions, tissue_names, output_dir)
    plot_best_worst_predictions(predictions, true_proportions, tissue_names,
                                best_indices, best_mae, worst_indices, worst_mae,
                                output_dir)
    
    if df_mixture_type is not None:
        plot_mixture_type_analysis(predictions, true_proportions, df_mixture_type, output_dir)
    
    # Generate reports
    generate_text_report(metrics, df_per_tissue, df_mixture_type,
                        best_indices, worst_indices, output_dir, phase)
    
    # Save all results
    save_results(metrics, df_per_tissue, df_mixture_type, predictions,
                true_proportions, output_dir)
    
    # Final summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nKey metrics:")
    print(f"  MAE: {metrics['mae']:.4f} ({metrics['mae']*100:.2f}%)")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Pearson r: {metrics['pearson_r']:.4f}")
    print("\nGenerated files:")
    print(f"  - evaluation_report.txt")
    print(f"  - metrics.json")
    print(f"  - per_tissue_metrics.csv")
    print(f"  - mixture_type_metrics.csv (if applicable)")
    print(f"  - predictions.npy, true_proportions.npy")
    print(f"  - figures/ (10 detailed plots)")
    print("="*80)


if __name__ == '__main__':
    main()
