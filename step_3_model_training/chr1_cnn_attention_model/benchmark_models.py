#!/usr/bin/env python3
"""
Benchmark: Compare Aggregation Model vs CNN+Attention Model
============================================================

Loads trained checkpoints and evaluates both models on test set.
"""

import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from dataloader_filelevel import create_filelevel_dataloaders
from model_methylation_aggregation import TissueBERT as AggregationModel
from model_cnn_attention import CNNAttentionTissueBERT as CNNAttentionModel


def load_model(checkpoint_path, model_class, config):
    """Load trained model from checkpoint"""
    cfg = config['model']
    
    model = model_class(
        vocab_size=cfg['vocab_size'],
        hidden_size=cfg['hidden_size'],
        num_hidden_layers=cfg['num_layers'],
        num_attention_heads=cfg['num_attention_heads'],
        intermediate_size=cfg['intermediate_size'],
        max_position_embeddings=cfg['max_seq_length'],
        num_classes=cfg['num_classes'],
        dropout=cfg['dropout']
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def evaluate_model(model, dataloader, device):
    """Evaluate model on dataset"""
    model.eval()
    model.to(device)
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            dna = batch['dna_tokens'].to(device)
            meth = batch['methylation'].to(device)
            targets = batch['tissue_label'].to(device)
            
            outputs = model(dna, meth)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = (all_preds == all_targets).mean()
    
    # Top-3 accuracy
    top3_preds = np.argsort(all_probs, axis=1)[:, -3:]
    top3_acc = np.array([target in top3 for target, top3 in zip(all_targets, top3_preds)]).mean()
    
    # Per-class accuracy
    per_class_acc = {}
    for cls in np.unique(all_targets):
        mask = all_targets == cls
        if mask.sum() > 0:
            per_class_acc[int(cls)] = (all_preds[mask] == all_targets[mask]).mean()
    
    return {
        'accuracy': accuracy,
        'top3_accuracy': top3_acc,
        'per_class_accuracy': per_class_acc,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aggregation_checkpoint', type=str, required=True,
                       help='Path to aggregation model checkpoint')
    parser.add_argument('--cnn_checkpoint', type=str, required=True,
                       help='Path to CNN+Attention model checkpoint')
    parser.add_argument('--aggregation_config', type=str, required=True,
                       help='Path to aggregation model config')
    parser.add_argument('--cnn_config', type=str, required=True,
                       help='Path to CNN+Attention model config')
    parser.add_argument('--output_dir', type=str, default='./benchmark_results',
                       help='Output directory for results')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load configs
    with open(args.aggregation_config) as f:
        agg_config = yaml.safe_load(f)
    with open(args.cnn_config) as f:
        cnn_config = yaml.safe_load(f)
    
    # Create test dataloader (use config from CNN model)
    print("Creating test dataloader...")
    _, _, test_loader = create_filelevel_dataloaders(
        hdf5_path=cnn_config['data']['hdf5_path'],
        train_csv=cnn_config['data']['train_csv'],
        val_csv=cnn_config['data']['val_csv'],
        test_csv=cnn_config['data']['test_csv'],
        batch_size=8,
        num_workers=4,
        pin_memory=True
    )
    print(f"Test batches: {len(test_loader)}\n")
    
    # Load models
    print("="*80)
    print("Loading Aggregation Model...")
    print("="*80)
    agg_model = load_model(args.aggregation_checkpoint, AggregationModel, agg_config)
    agg_params = sum(p.numel() for p in agg_model.parameters())
    print(f"Parameters: {agg_params:,}\n")
    
    print("="*80)
    print("Loading CNN+Attention Model...")
    print("="*80)
    cnn_model = load_model(args.cnn_checkpoint, CNNAttentionModel, cnn_config)
    cnn_params = sum(p.numel() for p in cnn_model.parameters())
    print(f"Parameters: {cnn_params:,}\n")
    
    # Evaluate aggregation model
    print("="*80)
    print("Evaluating Aggregation Model")
    print("="*80)
    agg_results = evaluate_model(agg_model, test_loader, device)
    print(f"Accuracy: {agg_results['accuracy']:.4f}")
    print(f"Top-3 Accuracy: {agg_results['top3_accuracy']:.4f}\n")
    
    # Evaluate CNN+Attention model
    print("="*80)
    print("Evaluating CNN+Attention Model")
    print("="*80)
    cnn_results = evaluate_model(cnn_model, test_loader, device)
    print(f"Accuracy: {cnn_results['accuracy']:.4f}")
    print(f"Top-3 Accuracy: {cnn_results['top3_accuracy']:.4f}\n")
    
    # Compare
    print("="*80)
    print("COMPARISON")
    print("="*80)
    print(f"\nOverall Accuracy:")
    print(f"  Aggregation: {agg_results['accuracy']:.4f}")
    print(f"  CNN+Attention: {cnn_results['accuracy']:.4f}")
    print(f"  Improvement: {(cnn_results['accuracy'] - agg_results['accuracy'])*100:.2f}%")
    
    print(f"\nTop-3 Accuracy:")
    print(f"  Aggregation: {agg_results['top3_accuracy']:.4f}")
    print(f"  CNN+Attention: {cnn_results['top3_accuracy']:.4f}")
    print(f"  Improvement: {(cnn_results['top3_accuracy'] - agg_results['top3_accuracy'])*100:.2f}%")
    
    print(f"\nModel Size:")
    print(f"  Aggregation: {agg_params:,} parameters")
    print(f"  CNN+Attention: {cnn_params:,} parameters")
    print(f"  Ratio: {cnn_params/agg_params:.2f}x")
    
    # Per-class comparison
    print(f"\nPer-Class Accuracy Comparison:")
    all_classes = set(agg_results['per_class_accuracy'].keys()) | set(cnn_results['per_class_accuracy'].keys())
    
    improvements = []
    for cls in sorted(all_classes):
        agg_acc = agg_results['per_class_accuracy'].get(cls, 0.0)
        cnn_acc = cnn_results['per_class_accuracy'].get(cls, 0.0)
        improvement = cnn_acc - agg_acc
        improvements.append(improvement)
        print(f"  Class {cls:2d}: Agg={agg_acc:.3f}, CNN={cnn_acc:.3f}, Δ={improvement:+.3f}")
    
    print(f"\nAverage per-class improvement: {np.mean(improvements)*100:.2f}%")
    
    # Save results
    results_summary = {
        'aggregation': {
            'accuracy': float(agg_results['accuracy']),
            'top3_accuracy': float(agg_results['top3_accuracy']),
            'parameters': agg_params,
            'per_class_accuracy': {int(k): float(v) for k, v in agg_results['per_class_accuracy'].items()}
        },
        'cnn_attention': {
            'accuracy': float(cnn_results['accuracy']),
            'top3_accuracy': float(cnn_results['top3_accuracy']),
            'parameters': cnn_params,
            'per_class_accuracy': {int(k): float(v) for k, v in cnn_results['per_class_accuracy'].items()}
        },
        'improvement': {
            'accuracy': float(cnn_results['accuracy'] - agg_results['accuracy']),
            'top3_accuracy': float(cnn_results['top3_accuracy'] - agg_results['top3_accuracy']),
            'avg_per_class': float(np.mean(improvements))
        }
    }
    
    import json
    with open(output_dir / 'benchmark_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir / 'benchmark_results.json'}")
    
    # Determine winner
    print("\n" + "="*80)
    if cnn_results['accuracy'] > agg_results['accuracy']:
        print("🏆 WINNER: CNN+Attention Model")
        print(f"   +{(cnn_results['accuracy'] - agg_results['accuracy'])*100:.2f}% accuracy improvement")
    elif cnn_results['accuracy'] < agg_results['accuracy']:
        print("🏆 WINNER: Aggregation Model")
        print(f"   +{(agg_results['accuracy'] - cnn_results['accuracy'])*100:.2f}% accuracy advantage")
    else:
        print("🤝 TIE: Both models perform equally")
    print("="*80)


if __name__ == '__main__':
    main()
