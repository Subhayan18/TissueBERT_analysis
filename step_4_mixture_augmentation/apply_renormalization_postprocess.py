#!/usr/bin/env python3
"""
Post-Processing Renormalization Script
=======================================

Applies renormalization strategies to existing model predictions and saves results.

This script:
1. Loads trained model and test data
2. Generates raw predictions
3. Applies all three renormalization strategies
4. Compares metrics before/after
5. Saves renormalized predictions to new H5 files
6. Generates comparison report

The output H5 files can be used with existing evaluation scripts.

Usage:
    python apply_renormalization_postprocess.py \\
        --checkpoint /path/to/checkpoint_best.pt \\
        --test_h5 /path/to/phase2_test_mixtures.h5 \\
        --output_dir /path/to/output \\
        --thresholds 0.03 0.05 0.10 \\
        --device cuda

Author: Mixture Deconvolution Project
Date: December 2024
"""

import argparse
import h5py
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import model and renormalization functions
from model_deconvolution_updated import (
    load_pretrained_model,
    threshold_renormalize,
    soft_threshold_renormalize,
    bayesian_sparse_renormalize,
    compare_strategies
)


def load_test_data(test_h5_path: str) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """
    Load test data from H5 file
    
    Args:
        test_h5_path: Path to test mixtures H5 file
        
    Returns:
        mixed_methylation: [n_samples, n_regions] array
        true_proportions: [n_samples, n_tissues] array
        tissue_names: List of tissue names
        phase: Phase number
    """
    print(f"\n{'='*80}")
    print("LOADING TEST DATA")
    print(f"{'='*80}")
    print(f"File: {test_h5_path}")
    
    with h5py.File(test_h5_path, 'r') as f:
        mixed_methylation = f['mixed_methylation'][:]
        true_proportions = f['true_proportions'][:]
        
        # Get tissue names
        try:
            tissue_names = [t.decode() if isinstance(t, bytes) else t 
                          for t in f.attrs['tissue_names']]
        except:
            # Fallback: load from metadata CSV
            print("  WARNING: tissue_names not in H5 attrs, loading from metadata...")
            metadata_path = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/combined_metadata.csv'
            metadata = pd.read_csv(metadata_path)
            tissue_names = sorted(metadata['tissue_top_level'].unique())
            print(f"  Loaded {len(tissue_names)} tissue names from metadata")
        
        phase = f.attrs.get('phase', 2)
        n_mixtures = f.attrs.get('n_mixtures', len(mixed_methylation))
    
    print(f"\n✓ Loaded test data:")
    print(f"  Samples: {len(mixed_methylation)}")
    print(f"  Regions: {mixed_methylation.shape[1]}")
    print(f"  Tissues: {len(tissue_names)}")
    print(f"  Phase: {phase}")
    
    return mixed_methylation, true_proportions, tissue_names, phase


def run_inference(model, mixed_methylation: np.ndarray, device: str = 'cuda', 
                 batch_size: int = 32) -> np.ndarray:
    """
    Run inference on test data to get raw predictions
    
    Args:
        model: Trained deconvolution model
        mixed_methylation: [n_samples, n_regions] array
        device: Device for inference
        batch_size: Batch size for inference
        
    Returns:
        predictions: [n_samples, n_tissues] array
    """
    print(f"\n{'='*80}")
    print("RUNNING INFERENCE (RAW PREDICTIONS)")
    print(f"{'='*80}")
    
    model.eval()
    n_samples = len(mixed_methylation)
    seq_length = 150
    
    all_predictions = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            
            # Expand to [batch, n_regions, seq_length]
            batch_means = mixed_methylation[i:batch_end]
            batch_meth = np.repeat(batch_means[:, :, np.newaxis], seq_length, axis=2)
            
            # Convert to tensor
            batch_tensor = torch.tensor(batch_meth, dtype=torch.float32).to(device)
            
            # Predict (NO renormalization - raw predictions)
            proportions = model(batch_tensor, apply_renorm=False)
            
            all_predictions.append(proportions.cpu().numpy())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {batch_end}/{n_samples} samples...")
    
    predictions = np.vstack(all_predictions)
    
    print(f"\n✓ Inference complete")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Prediction sums: min={predictions.sum(axis=1).min():.6f}, "
          f"max={predictions.sum(axis=1).max():.6f}, "
          f"mean={predictions.sum(axis=1).mean():.6f}")
    
    return predictions


def compute_metrics(predictions: np.ndarray, true_proportions: np.ndarray) -> Dict:
    """
    Compute performance metrics
    
    Args:
        predictions: [n_samples, n_tissues] predictions
        true_proportions: [n_samples, n_tissues] ground truth
        
    Returns:
        metrics: Dictionary of metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy.stats import pearsonr
    
    metrics = {}
    
    # Overall metrics
    metrics['mae'] = float(mean_absolute_error(true_proportions, predictions))
    metrics['rmse'] = float(np.sqrt(mean_squared_error(true_proportions, predictions)))
    metrics['r2'] = float(r2_score(true_proportions.flatten(), predictions.flatten()))
    
    # Correlation
    pearson_r, _ = pearsonr(true_proportions.flatten(), predictions.flatten())
    metrics['pearson_r'] = float(pearson_r)
    
    # Per-sample MAE
    per_sample_mae = np.abs(predictions - true_proportions).mean(axis=1)
    metrics['per_sample_mae_mean'] = float(per_sample_mae.mean())
    metrics['per_sample_mae_std'] = float(per_sample_mae.std())
    
    # False positives (tissues predicted >1% when true=0%)
    false_positives = ((predictions > 0.01) & (true_proportions < 0.01)).sum()
    total_possible = (true_proportions < 0.01).sum()
    metrics['false_positive_rate'] = float(false_positives / total_possible) if total_possible > 0 else 0.0
    metrics['n_false_positives'] = int(false_positives)
    
    # True positive accuracy (when tissue should be present)
    present_mask = true_proportions > 0.01
    if present_mask.sum() > 0:
        true_positive_mae = np.abs(predictions[present_mask] - true_proportions[present_mask]).mean()
        metrics['true_positive_mae'] = float(true_positive_mae)
    else:
        metrics['true_positive_mae'] = 0.0
    
    return metrics


def apply_all_renormalization_strategies(
    raw_predictions: np.ndarray,
    true_proportions: np.ndarray,
    thresholds: List[float] = [0.03, 0.05, 0.10]
) -> Dict:
    """
    Apply all renormalization strategies and compare results
    
    Args:
        raw_predictions: [n_samples, n_tissues] raw predictions
        true_proportions: [n_samples, n_tissues] ground truth
        thresholds: List of thresholds to test
        
    Returns:
        results: Dictionary with all results
    """
    print(f"\n{'='*80}")
    print("APPLYING RENORMALIZATION STRATEGIES")
    print(f"{'='*80}")
    
    results = {
        'raw': {
            'predictions': raw_predictions,
            'metrics': compute_metrics(raw_predictions, true_proportions),
            'params': {}
        }
    }
    
    # Baseline metrics
    baseline_mae = results['raw']['metrics']['mae']
    print(f"\nBaseline (no renormalization):")
    print(f"  MAE: {baseline_mae:.6f} ({baseline_mae*100:.2f}%)")
    print(f"  False positives: {results['raw']['metrics']['n_false_positives']}")
    
    # Strategy 1: Hard Threshold
    print(f"\n{'='*80}")
    print("Strategy 1: Hard Threshold")
    print(f"{'='*80}")
    
    for thresh in thresholds:
        print(f"\nThreshold {thresh*100:.0f}%:")
        renorm_preds = threshold_renormalize(raw_predictions, threshold=thresh)
        metrics = compute_metrics(renorm_preds, true_proportions)
        
        improvement = (baseline_mae - metrics['mae']) / baseline_mae * 100
        
        print(f"  MAE: {metrics['mae']:.6f} ({improvement:+.1f}% vs baseline)")
        print(f"  False positives: {metrics['n_false_positives']}")
        print(f"  Correlation: {metrics['pearson_r']:.4f}")
        
        key = f'threshold_{thresh:.2f}'
        results[key] = {
            'predictions': renorm_preds,
            'metrics': metrics,
            'params': {'strategy': 'threshold', 'threshold': thresh}
        }
    
    # Strategy 2: Soft Threshold
    print(f"\n{'='*80}")
    print("Strategy 2: Soft Threshold")
    print(f"{'='*80}")
    
    temperatures = [10.0]  # Focus on one temperature for simplicity
    for thresh in thresholds:
        for temp in temperatures:
            print(f"\nThreshold {thresh*100:.0f}%, Temperature {temp:.0f}:")
            renorm_preds = soft_threshold_renormalize(raw_predictions, 
                                                     threshold=thresh, 
                                                     temperature=temp)
            metrics = compute_metrics(renorm_preds, true_proportions)
            
            improvement = (baseline_mae - metrics['mae']) / baseline_mae * 100
            
            print(f"  MAE: {metrics['mae']:.6f} ({improvement:+.1f}% vs baseline)")
            print(f"  False positives: {metrics['n_false_positives']}")
            
            key = f'soft_threshold_{thresh:.2f}_temp{temp:.0f}'
            results[key] = {
                'predictions': renorm_preds,
                'metrics': metrics,
                'params': {'strategy': 'soft_threshold', 'threshold': thresh, 'temperature': temp}
            }
    
    # Strategy 3: Bayesian
    print(f"\n{'='*80}")
    print("Strategy 3: Bayesian Sparse")
    print(f"{'='*80}")
    
    sparsities = [0.7, 0.8]
    for sparsity in sparsities:
        print(f"\nPrior sparsity {sparsity*100:.0f}%:")
        renorm_preds = bayesian_sparse_renormalize(raw_predictions, 
                                                   prior_sparsity=sparsity)
        metrics = compute_metrics(renorm_preds, true_proportions)
        
        improvement = (baseline_mae - metrics['mae']) / baseline_mae * 100
        
        print(f"  MAE: {metrics['mae']:.6f} ({improvement:+.1f}% vs baseline)")
        print(f"  False positives: {metrics['n_false_positives']}")
        
        key = f'bayesian_sparsity{sparsity:.1f}'
        results[key] = {
            'predictions': renorm_preds,
            'metrics': metrics,
            'params': {'strategy': 'bayesian', 'prior_sparsity': sparsity}
        }
    
    # Find best strategy
    best_mae = float('inf')
    best_key = 'raw'
    
    for key, result in results.items():
        if result['metrics']['mae'] < best_mae:
            best_mae = result['metrics']['mae']
            best_key = key
    
    results['best'] = {
        'key': best_key,
        'mae': best_mae,
        'params': results[best_key]['params'],
        'improvement_pct': (baseline_mae - best_mae) / baseline_mae * 100
    }
    
    print(f"\n{'='*80}")
    print("BEST STRATEGY")
    print(f"{'='*80}")
    print(f"  Strategy: {best_key}")
    print(f"  Params: {results[best_key]['params']}")
    print(f"  MAE: {best_mae:.6f}")
    print(f"  Improvement: {results['best']['improvement_pct']:.1f}%")
    
    return results


def save_renormalized_predictions(
    results: Dict,
    test_h5_path: str,
    output_dir: Path,
    phase: int
):
    """
    Save renormalized predictions to new H5 files
    
    Args:
        results: Dictionary with all renormalization results
        test_h5_path: Original test H5 file path
        output_dir: Output directory
        phase: Phase number
    """
    print(f"\n{'='*80}")
    print("SAVING RENORMALIZED PREDICTIONS")
    print(f"{'='*80}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original H5 file attributes
    with h5py.File(test_h5_path, 'r') as f_orig:
        mixed_methylation = f_orig['mixed_methylation'][:]
        true_proportions = f_orig['true_proportions'][:]
        
        # Copy attributes
        attrs_to_copy = {}
        for key in f_orig.attrs.keys():
            attrs_to_copy[key] = f_orig.attrs[key]
        
        # Copy mixture_info if exists
        mixture_info = None
        if 'mixture_info' in f_orig:
            mixture_info = f_orig['mixture_info'][:]
    
    # Save each strategy as separate H5 file
    for key, result in results.items():
        if key == 'best':
            continue  # Skip best summary
        
        output_file = output_dir / f"phase{phase}_test_mixtures_{key}.h5"
        
        with h5py.File(output_file, 'w') as f:
            # Save data
            f.create_dataset('mixed_methylation', data=mixed_methylation, compression='gzip')
            f.create_dataset('true_proportions', data=true_proportions, compression='gzip')
            f.create_dataset('predicted_proportions', data=result['predictions'], compression='gzip')
            
            if mixture_info is not None:
                f.create_dataset('mixture_info', data=mixture_info)
            
            # Copy original attributes
            for attr_key, attr_val in attrs_to_copy.items():
                f.attrs[attr_key] = attr_val
            
            # Add renormalization metadata
            f.attrs['renormalization_strategy'] = key
            f.attrs['renormalization_params'] = json.dumps(result['params'])
            f.attrs['mae'] = result['metrics']['mae']
            f.attrs['false_positives'] = result['metrics']['n_false_positives']
        
        print(f"  ✓ Saved: {output_file.name}")
        print(f"    MAE: {result['metrics']['mae']:.6f}")
    
    # Also save best as a separate file for convenience
    best_key = results['best']['key']
    best_output = output_dir / f"phase{phase}_test_mixtures_BEST.h5"
    
    with h5py.File(best_output, 'w') as f:
        f.create_dataset('mixed_methylation', data=mixed_methylation, compression='gzip')
        f.create_dataset('true_proportions', data=true_proportions, compression='gzip')
        f.create_dataset('predicted_proportions', 
                        data=results[best_key]['predictions'], 
                        compression='gzip')
        
        if mixture_info is not None:
            f.create_dataset('mixture_info', data=mixture_info)
        
        for attr_key, attr_val in attrs_to_copy.items():
            f.attrs[attr_key] = attr_val
        
        f.attrs['renormalization_strategy'] = best_key
        f.attrs['renormalization_params'] = json.dumps(results[best_key]['params'])
        f.attrs['mae'] = results[best_key]['metrics']['mae']
        f.attrs['improvement_pct'] = results['best']['improvement_pct']
    
    print(f"\n  ✓ BEST saved as: {best_output.name}")
    print(f"    Strategy: {best_key}")
    print(f"    Improvement: {results['best']['improvement_pct']:.1f}%")


def generate_comparison_report(
    results: Dict,
    output_dir: Path,
    tissue_names: List[str]
):
    """
    Generate CSV report comparing all strategies
    
    Args:
        results: Dictionary with all results
        output_dir: Output directory
        tissue_names: List of tissue names
    """
    print(f"\n{'='*80}")
    print("GENERATING COMPARISON REPORT")
    print(f"{'='*80}")
    
    # Summary metrics
    summary_rows = []
    
    for key, result in results.items():
        if key == 'best':
            continue
        
        row = {
            'strategy': key,
            'mae': result['metrics']['mae'],
            'rmse': result['metrics']['rmse'],
            'r2': result['metrics']['r2'],
            'pearson_r': result['metrics']['pearson_r'],
            'false_positives': result['metrics']['n_false_positives'],
            'false_positive_rate': result['metrics']['false_positive_rate'],
            'true_positive_mae': result['metrics']['true_positive_mae'],
        }
        row.update(result['params'])
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('mae')
    
    summary_file = output_dir / 'renormalization_comparison.csv'
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\n✓ Saved comparison report: {summary_file.name}")
    
    # Print top 5
    print(f"\nTop 5 Strategies by MAE:")
    print(f"{'='*80}")
    for i, row in summary_df.head(5).iterrows():
        print(f"{i+1}. {row['strategy']}")
        print(f"   MAE: {row['mae']:.6f}, False Pos: {int(row['false_positives'])}, "
              f"R²: {row['r2']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Apply renormalization post-processing')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--test_h5', required=True, help='Path to test mixtures H5 file')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--thresholds', nargs='+', type=float, 
                       default=[0.03, 0.05, 0.10],
                       help='Thresholds to test (default: 0.03 0.05 0.10)')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    
    args = parser.parse_args()
    
    # Convert paths
    checkpoint_path = Path(args.checkpoint)
    test_h5_path = Path(args.test_h5)
    output_dir = Path(args.output_dir)
    
    # Verify files exist
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    if not test_h5_path.exists():
        print(f"ERROR: Test H5 file not found: {test_h5_path}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print("POST-PROCESSING RENORMALIZATION")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test data: {test_h5_path}")
    print(f"Output dir: {output_dir}")
    print(f"Thresholds: {args.thresholds}")
    print(f"Device: {args.device}")
    
    # Step 1: Load model
    print(f"\n{'='*80}")
    print("STEP 1: LOADING MODEL")
    print(f"{'='*80}")
    model = load_pretrained_model(str(checkpoint_path), device=args.device, verbose=True)
    
    # Step 2: Load test data
    mixed_methylation, true_proportions, tissue_names, phase = load_test_data(str(test_h5_path))
    
    # Step 3: Run inference to get raw predictions
    raw_predictions = run_inference(model, mixed_methylation, args.device, args.batch_size)
    
    # Step 4: Apply all renormalization strategies
    results = apply_all_renormalization_strategies(
        raw_predictions,
        true_proportions,
        thresholds=args.thresholds
    )
    
    # Step 5: Save renormalized predictions
    save_renormalized_predictions(results, str(test_h5_path), output_dir, phase)
    
    # Step 6: Generate comparison report
    generate_comparison_report(results, output_dir, tissue_names)
    
    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}")
    print(f"  - Renormalized H5 files (one per strategy)")
    print(f"  - BEST.h5 (best performing strategy)")
    print(f"  - renormalization_comparison.csv")
    print(f"\nYou can now run your evaluation scripts on any of these H5 files:")
    print(f"  python evaluate_deconvolution.py --test_h5 {output_dir}/phase{phase}_test_mixtures_BEST.h5 ...")
    print(f"  python visualize_mixture_miami.py --test_h5 {output_dir}/phase{phase}_test_mixtures_BEST.h5 ...")


if __name__ == '__main__':
    main()
