#!/usr/bin/env python3
"""
Analyze Training Results
Comprehensive analysis of TissueBERT training logs and performance

This script can be run after training to generate detailed analysis reports
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


def analyze_training_log(log_dir):
    """Analyze training log CSV file"""
    log_file = Path(log_dir) / 'training_log.csv'
    
    if not log_file.exists():
        print(f"❌ Training log not found: {log_file}")
        return None
    
    df = pd.read_csv(log_file)
    
    print("\n" + "="*80)
    print("TRAINING LOG ANALYSIS")
    print("="*80)
    
    print(f"\nBasic Statistics:")
    print(f"  Total epochs: {df['epoch'].max()}")
    print(f"  Total steps: {df['step'].max():,}")
    print(f"  Training time: {df['time_elapsed'].max() / 3600:.2f} hours")
    
    print(f"\nFinal Metrics (Last Epoch):")
    last_row = df.iloc[-1]
    print(f"  Train Loss: {last_row['train_loss']:.4f}")
    print(f"  Train Acc: {last_row['train_acc']:.4f}")
    print(f"  Val Loss: {last_row['val_loss']:.4f}")
    print(f"  Val Acc: {last_row['val_acc']:.4f}")
    print(f"  Learning Rate: {last_row['learning_rate']:.2e}")
    
    print(f"\nBest Metrics:")
    best_val_loss_idx = df['val_loss'].idxmin()
    best_val_acc_idx = df['val_acc'].idxmax()
    
    print(f"  Best Val Loss: {df.loc[best_val_loss_idx, 'val_loss']:.4f} (epoch {df.loc[best_val_loss_idx, 'epoch']})")
    print(f"  Best Val Acc: {df.loc[best_val_acc_idx, 'val_acc']:.4f} (epoch {df.loc[best_val_acc_idx, 'epoch']})")
    
    print(f"\nConvergence Analysis:")
    train_val_loss_gap = abs(last_row['train_loss'] - last_row['val_loss'])
    train_val_acc_gap = abs(last_row['train_acc'] - last_row['val_acc'])
    print(f"  Train-Val Loss Gap: {train_val_loss_gap:.4f}")
    print(f"  Train-Val Acc Gap: {train_val_acc_gap:.4f}")
    
    if train_val_loss_gap < 0.1:
        print(f"  ✓ Good convergence (gap < 0.1)")
    elif train_val_loss_gap < 0.2:
        print(f"  ⚠️  Moderate overfitting (0.1 < gap < 0.2)")
    else:
        print(f"  ❌ Significant overfitting (gap > 0.2)")
    
    print(f"\nGPU Utilization:")
    print(f"  Peak Memory: {df['gpu_memory'].max():.2f} GB")
    print(f"  Average Memory: {df['gpu_memory'].mean():.2f} GB")
    
    return df


def analyze_tissue_metrics(log_dir, num_classes=22):
    """Analyze per-tissue performance metrics"""
    tissue_file = Path(log_dir) / 'per_tissue_metrics.csv'
    
    if not tissue_file.exists():
        print(f"❌ Per-tissue metrics not found: {tissue_file}")
        return None
    
    df = pd.read_csv(tissue_file)
    
    print("\n" + "="*80)
    print("PER-TISSUE PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Get last epoch metrics
    last_epoch = df['epoch'].max()
    last_epoch_df = df[df['epoch'] == last_epoch]
    
    print(f"\nFinal Epoch ({last_epoch}) Per-Tissue Performance:")
    print(f"{'Tissue':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-"*80)
    
    for _, row in last_epoch_df.iterrows():
        print(f"{int(row['tissue_id']):<8} {row['accuracy']:<10.4f} {row['precision']:<10.4f} "
              f"{row['recall']:<10.4f} {row['f1']:<10.4f}")
    
    # Summary statistics
    print(f"\nSummary Statistics (F1 Score):")
    print(f"  Mean: {last_epoch_df['f1'].mean():.4f}")
    print(f"  Median: {last_epoch_df['f1'].median():.4f}")
    print(f"  Std: {last_epoch_df['f1'].std():.4f}")
    print(f"  Min: {last_epoch_df['f1'].min():.4f} (Tissue {last_epoch_df.loc[last_epoch_df['f1'].idxmin(), 'tissue_id']})")
    print(f"  Max: {last_epoch_df['f1'].max():.4f} (Tissue {last_epoch_df.loc[last_epoch_df['f1'].idxmax(), 'tissue_id']})")
    
    # Identify problematic tissues
    print(f"\nWorst Performing Tissues (F1 < 0.7):")
    poor_tissues = last_epoch_df[last_epoch_df['f1'] < 0.7].sort_values('f1')
    if len(poor_tissues) > 0:
        for _, row in poor_tissues.iterrows():
            print(f"  Tissue {int(row['tissue_id'])}: F1 = {row['f1']:.4f}")
    else:
        print(f"  ✓ All tissues have F1 >= 0.7")
    
    # Best performing tissues
    print(f"\nBest Performing Tissues (F1 > 0.9):")
    good_tissues = last_epoch_df[last_epoch_df['f1'] > 0.9].sort_values('f1', ascending=False)
    if len(good_tissues) > 0:
        for _, row in good_tissues.iterrows():
            print(f"  Tissue {int(row['tissue_id'])}: F1 = {row['f1']:.4f}")
    else:
        print(f"  No tissues have F1 > 0.9 yet")
    
    return df


def analyze_final_results(results_dir):
    """Analyze final test results"""
    results_file = Path(results_dir) / 'final_results.json'
    
    if not results_file.exists():
        print(f"❌ Final results not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("\n" + "="*80)
    print("FINAL TEST SET RESULTS")
    print("="*80)
    
    print(f"\nOverall Performance:")
    print(f"  Test Loss: {results['test_loss']:.4f}")
    print(f"  Test Accuracy: {results['test_acc']:.4f}")
    print(f"  Top-3 Accuracy: {results['test_metrics']['top3_acc']:.4f}")
    
    print(f"\nAggregated Metrics:")
    print(f"  Macro Precision: {results['test_metrics']['macro_precision']:.4f}")
    print(f"  Macro Recall: {results['test_metrics']['macro_recall']:.4f}")
    print(f"  Macro F1: {results['test_metrics']['macro_f1']:.4f}")
    print(f"  Weighted Precision: {results['test_metrics']['weighted_precision']:.4f}")
    print(f"  Weighted Recall: {results['test_metrics']['weighted_recall']:.4f}")
    print(f"  Weighted F1: {results['test_metrics']['weighted_f1']:.4f}")
    
    return results


def create_summary_report(log_dir, results_dir, output_path):
    """Create comprehensive summary report"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TissueBERT Training - Comprehensive Analysis Report\n")
        f.write("="*80 + "\n\n")
        
        # Training log analysis
        train_df = analyze_training_log(log_dir)
        
        # Per-tissue metrics
        tissue_df = analyze_tissue_metrics(log_dir)
        
        # Final results
        results = analyze_final_results(results_dir)
        
        # Recommendations
        f.write("\n" + "="*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")
        
        if results and train_df is not None:
            test_acc = results['test_acc']
            val_acc = train_df.iloc[-1]['val_acc']
            
            if test_acc >= 0.90:
                f.write("✓ Excellent performance! Consider:\n")
                f.write("  - Using this model for inference\n")
                f.write("  - Testing on real cfDNA mixtures\n")
                f.write("  - Generating deconvolution results\n")
            elif test_acc >= 0.85:
                f.write("✓ Good performance! Consider:\n")
                f.write("  - Training for more epochs if not converged\n")
                f.write("  - Analyzing per-tissue errors\n")
                f.write("  - Testing on validation samples\n")
            else:
                f.write("⚠️  Performance needs improvement. Consider:\n")
                f.write("  - Training for more epochs\n")
                f.write("  - Adjusting hyperparameters (learning rate, batch size)\n")
                f.write("  - Analyzing data quality\n")
                f.write("  - Checking for class imbalance issues\n")
            
            f.write(f"\n  Train-Val-Test Consistency:\n")
            train_acc = train_df.iloc[-1]['train_acc']
            f.write(f"    Train Acc: {train_acc:.4f}\n")
            f.write(f"    Val Acc:   {val_acc:.4f}\n")
            f.write(f"    Test Acc:  {test_acc:.4f}\n")
            
            if abs(val_acc - test_acc) < 0.05:
                f.write(f"    ✓ Good generalization (val-test gap < 0.05)\n")
            else:
                f.write(f"    ⚠️  Check data splits (val-test gap = {abs(val_acc - test_acc):.4f})\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("End of Report\n")
        f.write("="*80 + "\n")
    
    print(f"\n✓ Summary report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze TissueBERT training results')
    parser.add_argument('--log_dir', type=str, required=True, 
                       help='Directory containing training logs')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing results')
    parser.add_argument('--output', type=str, default='analysis_report.txt',
                       help='Output report path')
    
    args = parser.parse_args()
    
    # Run analyses
    print("\n" + "="*80)
    print("TissueBERT Training Results Analysis")
    print("="*80)
    
    # Training log
    train_df = analyze_training_log(args.log_dir)
    
    # Per-tissue metrics  
    tissue_df = analyze_tissue_metrics(args.log_dir)
    
    # Final results
    results = analyze_final_results(args.results_dir)
    
    # Create summary report
    create_summary_report(args.log_dir, args.results_dir, args.output)
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
