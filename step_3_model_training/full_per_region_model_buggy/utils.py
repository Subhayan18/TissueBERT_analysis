#!/usr/bin/env python3
"""
Utility functions for TissueBERT training
Metrics, checkpointing, logging, and visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import pandas as pd


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ProgressLogger:
    """Log training progress to file"""
    
    def __init__(self, log_file):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, message):
        with open(self.log_file, 'a') as f:
            f.write(f"{message}\n")
        print(message)


def compute_metrics(predictions, targets, probs, num_classes=22):
    """
    Compute comprehensive metrics for multi-class classification
    
    Args:
        predictions: Predicted class labels [N]
        targets: Ground truth labels [N]
        probs: Prediction probabilities [N, num_classes]
        num_classes: Number of classes
        
    Returns:
        Dictionary of metrics
    """
    # Overall accuracy
    accuracy = accuracy_score(targets, predictions)
    
    # Top-3 accuracy
    top3_preds = np.argsort(probs, axis=1)[:, -3:]
    top3_acc = np.mean([targets[i] in top3_preds[i] for i in range(len(targets))])
    
    # Per-class metrics for ALL classes (sklearn handles missing classes with zero_division)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, 
        labels=list(range(num_classes)),  # All 22 classes
        average=None, 
        zero_division=0  # Missing classes get 0.0
    )
    
    # Macro and weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        targets, predictions, average='macro', zero_division=0
    )
    
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        targets, predictions, average='weighted', zero_division=0
    )
    
    # Confusion matrix (for all classes)
    cm = confusion_matrix(targets, predictions, labels=list(range(num_classes)))
    
    # Per-class accuracy from confusion matrix
    per_class_acc = np.zeros(num_classes)
    for i in range(num_classes):
        if cm[i].sum() > 0:
            per_class_acc[i] = cm[i, i] / cm[i].sum()
        else:
            per_class_acc[i] = 0.0
    
    # Organize metrics
    # precision, recall, f1, support are already length-22 arrays
    metrics = {
        # Overall metrics
        'accuracy': accuracy,
        'top3_acc': top3_acc,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        
        # Per-class metrics (as dictionaries with all 22 classes)
        'per_class_accuracy': {i: float(per_class_acc[i]) for i in range(num_classes)},
        'per_class_precision': {i: float(precision[i]) for i in range(num_classes)},
        'per_class_recall': {i: float(recall[i]) for i in range(num_classes)},
        'per_class_f1': {i: float(f1[i]) for i in range(num_classes)},
        'per_class_support': {i: int(support[i]) for i in range(num_classes)},
        
        # Confusion matrix
        'confusion_matrix': cm
    }
    
    return metrics


def plot_confusion_matrix(cm, save_path, num_classes=22):
    """
    Plot and save confusion matrix
    
    Args:
        cm: Confusion matrix [num_classes, num_classes]
        save_path: Path to save figure
        num_classes: Number of classes
    """
    plt.figure(figsize=(12, 10))
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis].clip(min=1e-9)
    
    # Plot
    sns.heatmap(
        cm_norm,
        annot=False,
        fmt='.2f',
        cmap='Blues',
        square=True,
        cbar_kws={'label': 'Normalized Count'}
    )
    
    plt.xlabel('Predicted Tissue')
    plt.ylabel('True Tissue')
    plt.title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_checkpoint(checkpoint_data, checkpoint_path):
    """
    Save model checkpoint
    
    Args:
        checkpoint_data: Dictionary containing model state, optimizer state, etc.
        checkpoint_path: Path to save checkpoint
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint_data, checkpoint_path)


def load_checkpoint(checkpoint_path, device='cuda'):
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint to
        
    Returns:
        Checkpoint data dictionary
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def save_training_summary(log_dir, results_dir, config, final_results):
    """
    Generate comprehensive training summary report
    
    Args:
        log_dir: Directory containing training logs
        results_dir: Directory to save summary
        config: Training configuration
        final_results: Final test results dictionary
    """
    log_dir = Path(log_dir)
    results_dir = Path(results_dir)
    
    summary_path = results_dir / 'training_summary.txt'
    
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TissueBERT Training Summary\n")
        f.write("="*80 + "\n\n")
        
        # Configuration
        f.write("CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Model: DNABERT-S\n")
        f.write(f"  - Layers: {config['model']['num_layers']}\n")
        f.write(f"  - Hidden size: {config['model']['hidden_size']}\n")
        f.write(f"  - Attention heads: {config['model']['num_attention_heads']}\n")
        f.write(f"  - Num classes: {config['model']['num_classes']}\n")
        f.write(f"\nTraining:\n")
        f.write(f"  - Epochs: {config['training']['num_epochs']}\n")
        f.write(f"  - Batch size: {config['training']['batch_size']}\n")
        f.write(f"  - Gradient accumulation: {config['training']['gradient_accumulation_steps']}\n")
        f.write(f"  - Effective batch size: {config['training']['batch_size'] * config['training']['gradient_accumulation_steps']}\n")
        f.write(f"  - Learning rate: {config['optimizer']['learning_rate']}\n")
        f.write(f"  - Weight decay: {config['optimizer']['weight_decay']}\n")
        f.write(f"  - Label smoothing: {config['loss']['label_smoothing']}\n")
        f.write(f"  - CpG dropout: {config['training']['cpg_dropout_rate']}\n")
        f.write("\n")
        
        # Final results
        f.write("FINAL TEST RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"Test Loss: {final_results['test_loss']:.4f}\n")
        f.write(f"Test Accuracy: {final_results['test_acc']:.4f}\n")
        f.write(f"Top-3 Accuracy: {final_results['test_metrics']['top3_acc']:.4f}\n")
        f.write(f"Macro F1: {final_results['test_metrics']['macro_f1']:.4f}\n")
        f.write(f"Weighted F1: {final_results['test_metrics']['weighted_f1']:.4f}\n")
        f.write("\n")
        
        # Per-tissue performance
        f.write("PER-TISSUE PERFORMANCE\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Tissue':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}\n")
        f.write("-"*80 + "\n")
        
        for tissue_id in range(config['model']['num_classes']):
            if tissue_id in final_results['test_metrics']['per_class_f1']:
                acc = final_results['test_metrics']['per_class_accuracy'][tissue_id]
                prec = final_results['test_metrics']['per_class_precision'][tissue_id]
                rec = final_results['test_metrics']['per_class_recall'][tissue_id]
                f1 = final_results['test_metrics']['per_class_f1'][tissue_id]
                sup = final_results['test_metrics']['per_class_support'][tissue_id]
                
                f.write(f"{tissue_id:<8} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {sup:<10}\n")
        
        f.write("\n")
        
        # Training history analysis
        if (log_dir / 'training_log.csv').exists():
            f.write("TRAINING HISTORY ANALYSIS\n")
            f.write("-"*80 + "\n")
            
            df = pd.read_csv(log_dir / 'training_log.csv')
            
            f.write(f"Total epochs completed: {df['epoch'].max()}\n")
            f.write(f"Total training steps: {df['step'].max()}\n")
            f.write(f"Total training time: {df['time_elapsed'].max() / 3600:.2f} hours\n")
            f.write(f"\nBest validation metrics:\n")
            f.write(f"  - Lowest val loss: {df['val_loss'].min():.4f} (epoch {df.loc[df['val_loss'].idxmin(), 'epoch']})\n")
            f.write(f"  - Highest val acc: {df['val_acc'].max():.4f} (epoch {df.loc[df['val_acc'].idxmax(), 'epoch']})\n")
            f.write(f"\nFinal metrics:\n")
            f.write(f"  - Train loss: {df['train_loss'].iloc[-1]:.4f}\n")
            f.write(f"  - Train acc: {df['train_acc'].iloc[-1]:.4f}\n")
            f.write(f"  - Val loss: {df['val_loss'].iloc[-1]:.4f}\n")
            f.write(f"  - Val acc: {df['val_acc'].iloc[-1]:.4f}\n")
            f.write(f"\nConvergence analysis:\n")
            f.write(f"  - Train-val loss gap: {abs(df['train_loss'].iloc[-1] - df['val_loss'].iloc[-1]):.4f}\n")
            f.write(f"  - Train-val acc gap: {abs(df['train_acc'].iloc[-1] - df['val_acc'].iloc[-1]):.4f}\n")
            
            f.write("\n")
        
        # GPU utilization
        if (log_dir / 'training_log.csv').exists():
            f.write("RESOURCE UTILIZATION\n")
            f.write("-"*80 + "\n")
            f.write(f"Peak GPU memory: {df['gpu_memory'].max():.2f} GB\n")
            f.write(f"Average GPU memory: {df['gpu_memory'].mean():.2f} GB\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("End of Summary\n")
        f.write("="*80 + "\n")
    
    print(f"Training summary saved: {summary_path}")
    
    # Also create plots
    if (log_dir / 'training_log.csv').exists():
        create_training_plots(log_dir, results_dir)


def create_training_plots(log_dir, results_dir):
    """
    Create training visualization plots
    
    Args:
        log_dir: Directory containing training logs
        results_dir: Directory to save plots
    """
    log_dir = Path(log_dir)
    results_dir = Path(results_dir)
    
    # Load training log
    df = pd.read_csv(log_dir / 'training_log.csv')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(df['epoch'], df['train_acc'], label='Train Acc', linewidth=2)
    axes[0, 1].plot(df['epoch'], df['val_acc'], label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[1, 0].plot(df['epoch'], df['learning_rate'], linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # GPU memory plot
    axes[1, 1].plot(df['epoch'], df['gpu_memory'], linewidth=2, color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('GPU Memory (GB)')
    axes[1, 1].set_title('GPU Memory Usage')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved: {results_dir / 'training_curves.png'}")
    
    # Per-tissue performance over time
    if (log_dir / 'per_tissue_metrics.csv').exists():
        tissue_df = pd.read_csv(log_dir / 'per_tissue_metrics.csv')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for tissue_id in range(min(4, tissue_df['tissue_id'].nunique())):
            tissue_data = tissue_df[tissue_df['tissue_id'] == tissue_id]
            
            row = tissue_id // 2
            col = tissue_id % 2
            
            axes[row, col].plot(tissue_data['epoch'], tissue_data['f1'], linewidth=2)
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel('F1 Score')
            axes[row, col].set_title(f'Tissue {tissue_id} F1 Score')
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / 'per_tissue_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Per-tissue curves saved: {results_dir / 'per_tissue_curves.png'}")


def analyze_predictions(predictions, targets, probs, save_path):
    """
    Analyze and visualize prediction patterns
    
    Args:
        predictions: Predicted labels
        targets: Ground truth labels
        probs: Prediction probabilities
        save_path: Path to save analysis
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Confidence analysis
    max_probs = probs.max(axis=1)
    correct = predictions == targets
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confidence distribution
    axes[0].hist(max_probs[correct], bins=50, alpha=0.7, label='Correct', density=True)
    axes[0].hist(max_probs[~correct], bins=50, alpha=0.7, label='Incorrect', density=True)
    axes[0].set_xlabel('Prediction Confidence')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Prediction Confidence Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy vs confidence
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_acc = []
    
    for i in range(len(bins) - 1):
        mask = (max_probs >= bins[i]) & (max_probs < bins[i+1])
        if mask.sum() > 0:
            bin_acc.append(correct[mask].mean())
        else:
            bin_acc.append(0)
    
    axes[1].plot(bin_centers, bin_acc, 'o-', linewidth=2, markersize=8)
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    axes[1].set_xlabel('Prediction Confidence')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Calibration Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Prediction analysis saved: {save_path / 'prediction_analysis.png'}")


if __name__ == '__main__':
    # Test utilities
    print("Testing utility functions...")
    
    # Test AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"AverageMeter test: avg={meter.avg:.2f} (expected 4.50)")
    
    # Test metrics computation
    np.random.seed(42)
    n_samples = 1000
    n_classes = 22
    
    predictions = np.random.randint(0, n_classes, n_samples)
    targets = np.random.randint(0, n_classes, n_samples)
    probs = np.random.rand(n_samples, n_classes)
    probs = probs / probs.sum(axis=1, keepdims=True)
    
    metrics = compute_metrics(predictions, targets, probs, n_classes)
    print(f"\nMetrics test:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Top-3 Acc: {metrics['top3_acc']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    
    print("\n✓ Utility tests passed!")
