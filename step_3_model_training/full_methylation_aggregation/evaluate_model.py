#!/usr/bin/env python3
"""
Comprehensive Evaluation Script
================================

Generates all prediction results, figures, and analysis from a trained model.

Usage:
    python evaluate_model.py --checkpoint /path/to/checkpoint_best_acc.pt --config config_fullgenome.yaml
    
Outputs:
    results/
    ├── predictions/
    │   ├── test_predictions.csv           # Per-sample predictions
    │   ├── test_predictions_detailed.csv  # With probabilities
    │   └── misclassified_samples.csv      # Errors analysis
    ├── figures/
    │   ├── confusion_matrix.png
    │   ├── confusion_matrix_normalized.png
    │   ├── per_tissue_accuracy.png
    │   ├── per_tissue_f1.png
    │   ├── confidence_distribution.png
    │   └── error_analysis.png
    └── summary_report.txt
"""

import os
import argparse
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataloader_filelevel import create_filelevel_dataloaders
from model_methylation_aggregation import TissueBERT
from utils import compute_metrics


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, checkpoint_path, config_path, output_dir):
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.predictions_dir = self.output_dir / 'predictions'
        self.figures_dir = self.output_dir / 'figures'
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self.load_model()
        
        # Load data
        self.test_loader = self.load_test_data()
        
        print(f"\nOutput directory: {self.output_dir}")
        print(f"  - Predictions: {self.predictions_dir}")
        print(f"  - Figures: {self.figures_dir}")
    
    def load_model(self):
        """Load trained model from checkpoint"""
        print(f"\nLoading model from: {self.checkpoint_path}")
        
        # Create model
        model = TissueBERT(
            vocab_size=self.config['model']['vocab_size'],
            hidden_size=self.config['model']['hidden_size'],
            num_hidden_layers=self.config['model'].get('num_layers', 3),
            num_attention_heads=self.config['model'].get('num_attention_heads', 4),
            intermediate_size=self.config['model'].get('intermediate_size', 1024),
            max_position_embeddings=self.config['model']['max_seq_length'],
            num_classes=self.config['model']['num_classes'],
            dropout=self.config['model']['dropout']
        )
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
        
        return model
    
    def load_test_data(self):
        """Load test dataloader"""
        print("\nLoading test data...")
        
        _, _, test_loader = create_filelevel_dataloaders(
            hdf5_path=self.config['data']['hdf5_path'],
            train_csv=self.config['data']['train_csv'],
            val_csv=self.config['data']['val_csv'],
            test_csv=self.config['data']['test_csv'],
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )
        
        return test_loader
    
    @torch.no_grad()
    def get_predictions(self):
        """Get predictions for all test samples"""
        print("\nGenerating predictions...")
        
        all_preds = []
        all_targets = []
        all_probs = []
        all_file_indices = []
        
        for batch in tqdm(self.test_loader, desc="Predicting"):
            dna = batch['dna_tokens'].to(self.device)
            meth = batch['methylation'].to(self.device)
            targets = batch['tissue_label'].to(self.device)
            file_idx = batch['file_idx']
            
            # Forward pass
            outputs = self.model(dna, meth)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            # Collect results
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_file_indices.extend(file_idx.numpy())
        
        return {
            'predictions': np.array(all_preds),
            'targets': np.array(all_targets),
            'probabilities': np.array(all_probs),
            'file_indices': np.array(all_file_indices)
        }
    
    def save_predictions(self, results):
        """Save predictions to CSV files"""
        print("\nSaving predictions...")
        
        # Basic predictions
        pred_df = pd.DataFrame({
            'file_idx': results['file_indices'],
            'true_label': results['targets'],
            'predicted_label': results['predictions'],
            'correct': results['predictions'] == results['targets'],
            'max_probability': results['probabilities'].max(axis=1),
        })
        pred_df.to_csv(self.predictions_dir / 'test_predictions.csv', index=False)
        print(f"  Saved: test_predictions.csv")
        
        # Detailed predictions with all probabilities
        prob_df = pd.DataFrame(
            results['probabilities'],
            columns=[f'prob_class_{i}' for i in range(results['probabilities'].shape[1])]
        )
        detailed_df = pd.concat([pred_df, prob_df], axis=1)
        detailed_df.to_csv(self.predictions_dir / 'test_predictions_detailed.csv', index=False)
        print(f"  Saved: test_predictions_detailed.csv")
        
        # Misclassified samples
        errors = pred_df[~pred_df['correct']].copy()
        errors['confidence'] = errors.apply(
            lambda row: results['probabilities'][row.name, row['predicted_label']], 
            axis=1
        )
        errors = errors.sort_values('confidence', ascending=False)
        errors.to_csv(self.predictions_dir / 'misclassified_samples.csv', index=False)
        print(f"  Saved: misclassified_samples.csv ({len(errors)} errors)")
    
    def plot_confusion_matrix(self, results):
        """Plot confusion matrices"""
        print("\nGenerating confusion matrices...")
        
        cm = confusion_matrix(results['targets'], results['predictions'])
        
        # Absolute counts
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix (Absolute Counts)')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: confusion_matrix.png")
        
        # Normalized
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', cbar_kws={'label': 'Proportion'})
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix (Normalized)')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: confusion_matrix_normalized.png")
    
    def plot_per_tissue_metrics(self, results):
        """Plot per-tissue performance metrics"""
        print("\nGenerating per-tissue metrics plots...")
        
        num_classes = self.config['model']['num_classes']
        
        # Compute per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            results['targets'], 
            results['predictions'],
            labels=list(range(num_classes)),
            average=None,
            zero_division=0
        )
        
        # Per-class accuracy from confusion matrix
        cm = confusion_matrix(results['targets'], results['predictions'], 
                             labels=list(range(num_classes)))
        per_class_acc = np.zeros(num_classes)
        for i in range(num_classes):
            if cm[i].sum() > 0:
                per_class_acc[i] = cm[i, i] / cm[i].sum()
            else:
                per_class_acc[i] = 0.0
        
        # Plot accuracy
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(num_classes)
        bars = ax.bar(x, per_class_acc, color='steelblue', alpha=0.7)
        
        # Color code by performance
        for i, bar in enumerate(bars):
            if per_class_acc[i] >= 0.9:
                bar.set_color('green')
            elif per_class_acc[i] >= 0.7:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax.set_xlabel('Tissue Class')
        ax.set_ylabel('Accuracy')
        ax.set_title('Per-Tissue Accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels(x, rotation=45)
        ax.axhline(y=per_class_acc.mean(), color='black', linestyle='--', 
                   label=f'Mean: {per_class_acc.mean():.3f}')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'per_tissue_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: per_tissue_accuracy.png")
        
        # Plot F1 scores
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x, f1, color='steelblue', alpha=0.7)
        ax.set_xlabel('Tissue Class')
        ax.set_ylabel('F1 Score')
        ax.set_title('Per-Tissue F1 Score')
        ax.set_xticks(x)
        ax.set_xticklabels(x, rotation=45)
        ax.axhline(y=f1.mean(), color='black', linestyle='--', 
                   label=f'Mean: {f1.mean():.3f}')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'per_tissue_f1.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: per_tissue_f1.png")
    
    def plot_confidence_distribution(self, results):
        """Plot prediction confidence distributions"""
        print("\nGenerating confidence distribution plots...")
        
        max_probs = results['probabilities'].max(axis=1)
        correct = results['predictions'] == results['targets']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Overall distribution
        ax1.hist(max_probs, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Maximum Probability (Confidence)')
        ax1.set_ylabel('Count')
        ax1.set_title('Prediction Confidence Distribution')
        ax1.axvline(x=max_probs.mean(), color='red', linestyle='--', 
                    label=f'Mean: {max_probs.mean():.3f}')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Correct vs incorrect
        ax2.hist([max_probs[correct], max_probs[~correct]], 
                bins=30, label=['Correct', 'Incorrect'], 
                color=['green', 'red'], alpha=0.6, edgecolor='black')
        ax2.set_xlabel('Maximum Probability (Confidence)')
        ax2.set_ylabel('Count')
        ax2.set_title('Confidence: Correct vs Incorrect Predictions')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: confidence_distribution.png")
    
    def plot_error_analysis(self, results):
        """Analyze and plot errors"""
        print("\nGenerating error analysis plots...")
        
        cm = confusion_matrix(results['targets'], results['predictions'])
        
        # Find most confused pairs
        np.fill_diagonal(cm, 0)  # Remove correct predictions
        confused_pairs = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] > 0:
                    confused_pairs.append((i, j, cm[i, j]))
        
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        top_confused = confused_pairs[:10]
        
        if len(top_confused) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            labels = [f"{true}->{pred}" for true, pred, _ in top_confused]
            counts = [count for _, _, count in top_confused]
            
            ax.barh(labels, counts, color='coral', alpha=0.7)
            ax.set_xlabel('Number of Misclassifications')
            ax.set_ylabel('True -> Predicted')
            ax.set_title('Top 10 Most Confused Tissue Pairs')
            ax.invert_yaxis()
            
            for i, count in enumerate(counts):
                ax.text(count, i, f' {count}', va='center')
            
            plt.tight_layout()
            plt.savefig(self.figures_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: error_analysis.png")
    
    def save_summary_report(self, results):
        """Generate comprehensive summary report"""
        print("\nGenerating summary report...")
        
        # Compute metrics
        metrics = compute_metrics(
            results['predictions'],
            results['targets'],
            results['probabilities'],
            self.config['model']['num_classes']
        )
        
        report_path = self.output_dir / 'summary_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MODEL EVALUATION SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Overall metrics
            f.write("OVERALL PERFORMANCE\n")
            f.write("-"*80 + "\n")
            f.write(f"Test Accuracy:      {metrics['accuracy']:.4f}\n")
            f.write(f"Top-3 Accuracy:     {metrics['top3_acc']:.4f}\n")
            f.write(f"Macro Precision:    {metrics['macro_precision']:.4f}\n")
            f.write(f"Macro Recall:       {metrics['macro_recall']:.4f}\n")
            f.write(f"Macro F1:           {metrics['macro_f1']:.4f}\n")
            f.write(f"Weighted Precision: {metrics['weighted_precision']:.4f}\n")
            f.write(f"Weighted Recall:    {metrics['weighted_recall']:.4f}\n")
            f.write(f"Weighted F1:        {metrics['weighted_f1']:.4f}\n\n")
            
            # Per-class metrics
            f.write("PER-TISSUE PERFORMANCE\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Class':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Support':<10}\n")
            f.write("-"*80 + "\n")
            
            for i in range(self.config['model']['num_classes']):
                f.write(f"{i:<8} {metrics['per_class_accuracy'][i]:<8.3f} "
                       f"{metrics['per_class_precision'][i]:<8.3f} {metrics['per_class_recall'][i]:<8.3f} "
                       f"{metrics['per_class_f1'][i]:<8.3f} {metrics['per_class_support'][i]:<10}\n")
            
            f.write("\n")
            
            # Error analysis
            errors = results['predictions'] != results['targets']
            n_errors = errors.sum()
            
            f.write("ERROR ANALYSIS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total samples:      {len(results['targets'])}\n")
            f.write(f"Correct:            {(~errors).sum()} ({(~errors).sum()/len(results['targets'])*100:.2f}%)\n")
            f.write(f"Incorrect:          {n_errors} ({n_errors/len(results['targets'])*100:.2f}%)\n")
            
            if n_errors > 0:
                error_probs = results['probabilities'][errors].max(axis=1)
                f.write(f"Avg error confidence: {error_probs.mean():.4f}\n")
            
            f.write("\n")
            
            # Configuration
            f.write("MODEL CONFIGURATION\n")
            f.write("-"*80 + "\n")
            f.write(f"Checkpoint: {self.checkpoint_path}\n")
            f.write(f"Classes: {self.config['model']['num_classes']}\n")
            f.write(f"Hidden size: {self.config['model']['hidden_size']}\n")
            f.write(f"Dropout: {self.config['model']['dropout']}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"  Saved: summary_report.txt")
    
    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*80)
        
        # Get predictions
        results = self.get_predictions()
        
        # Save predictions
        self.save_predictions(results)
        
        # Generate figures
        self.plot_confusion_matrix(results)
        self.plot_per_tissue_metrics(results)
        self.plot_confidence_distribution(results)
        self.plot_error_analysis(results)
        
        # Save summary
        self.save_summary_report(results)
        
        # Print summary
        accuracy = (results['predictions'] == results['targets']).mean()
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"\nAll results saved to: {self.output_dir}")
        print(f"  - Predictions: {self.predictions_dir}")
        print(f"  - Figures: {self.figures_dir}")
        print(f"  - Summary: {self.output_dir / 'summary_report.txt'}")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Model Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (e.g., checkpoint_best_acc.pt)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results (default: auto-generated from checkpoint path)')
    
    args = parser.parse_args()
    
    # Auto-generate output directory if not provided
    if args.output is None:
        checkpoint_path = Path(args.checkpoint)
        # Get parent of checkpoints directory (e.g., fullgenome_results/)
        results_dir = checkpoint_path.parent.parent
        # Create evaluation_results subdirectory
        args.output = results_dir / 'evaluation_results'
        print(f"\nAuto-detected output directory: {args.output}")
    
    evaluator = ModelEvaluator(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output
    )
    
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()
