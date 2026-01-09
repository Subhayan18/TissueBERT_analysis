#!/usr/bin/env python3
"""
PDAC Clinical Prediction - Inference Pipeline (HDF5)
=====================================================

Run trained deconvolution model on PDAC samples to predict tissue proportions.

Input:  HDF5 file + metadata CSV from convert_pdac_beta_to_hdf5.py
Output: Tissue proportion predictions with clinical reports

Author: PDAC Clinical Prediction Pipeline
Date: January 2026
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import h5py
from tqdm import tqdm
import json
from datetime import datetime


# Import the model (make sure model_deconvolution.py is in the same directory or PYTHONPATH)
try:
    from model_deconvolution import TissueBERTDeconvolution, apply_renormalization
except ImportError:
    print("ERROR: Could not import model_deconvolution.py")
    print("Make sure model_deconvolution.py is in the same directory or PYTHONPATH")
    sys.exit(1)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run inference on PDAC samples for tissue deconvolution',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input_h5',
        type=str,
        required=True,
        help='HDF5 file containing methylation data'
    )
    
    parser.add_argument(
        '--input_metadata',
        type=str,
        required=True,
        help='Metadata CSV file with sample information'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pt file)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for predictions and reports'
    )
    
    parser.add_argument(
        '--tissue_names',
        type=str,
        default=None,
        help='Path to file with tissue names (one per line, 22 total). If not provided, uses generic names.'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for inference'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for inference'
    )
    
    parser.add_argument(
        '--apply_renormalization',
        action='store_true',
        help='Apply post-processing renormalization to predictions'
    )
    
    parser.add_argument(
        '--renorm_strategy',
        type=str,
        default='threshold',
        choices=['threshold', 'soft_threshold', 'bayesian', 'none'],
        help='Renormalization strategy to use'
    )
    
    parser.add_argument(
        '--renorm_threshold',
        type=float,
        default=0.05,
        help='Threshold for renormalization (tissues below this are zeroed)'
    )
    
    parser.add_argument(
        '--min_proportion',
        type=float,
        default=0.01,
        help='Minimum proportion to report in clinical output (1%% default)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    
    return parser.parse_args()


def load_tissue_names(tissue_names_path, num_classes=22):
    """Load tissue names from file or generate default names"""
    if tissue_names_path and Path(tissue_names_path).exists():
        with open(tissue_names_path, 'r') as f:
            tissue_names = [line.strip() for line in f if line.strip()]
        
        if len(tissue_names) != num_classes:
            print(f"⚠️  WARNING: Expected {num_classes} tissue names, got {len(tissue_names)}")
            print(f"    Using default names instead")
            tissue_names = [f"Tissue_{i:02d}" for i in range(num_classes)]
    else:
        # Default tissue names
        tissue_names = [f"Tissue_{i:02d}" for i in range(num_classes)]
    
    return tissue_names


def load_checkpoint(checkpoint_path, device='cuda', verbose=False):
    """
    Load trained model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint .pt file
        device: Device to load model on
        verbose: Print loading information
    
    Returns:
        model: Loaded model
        config: Model configuration
        checkpoint: Full checkpoint dict
    """
    if verbose:
        print(f"\nLoading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract configuration
    config = checkpoint['config']
    model_config = config['model']
    
    if verbose:
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Global step: {checkpoint['global_step']}")
        print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
        print(f"  Best val MAE: {checkpoint.get('best_val_mae', 'N/A')}")
        print(f"\nModel configuration:")
        print(f"  Regions: {model_config['n_regions']}")
        print(f"  Hidden size: {model_config['hidden_size']}")
        print(f"  Num classes: {model_config['num_classes']}")
        print(f"  Dropout: {model_config['dropout']}")
    
    # Initialize model
    model = TissueBERTDeconvolution(
        vocab_size=model_config.get('vocab_size', 69),  # Not used in this model version
        hidden_size=model_config['hidden_size'],
        num_hidden_layers=model_config.get('num_hidden_layers', 3),
        num_attention_heads=model_config.get('num_attention_heads', 4),
        intermediate_size=model_config['intermediate_size'],
        max_position_embeddings=model_config.get('max_position_embeddings', 150),
        num_classes=model_config['num_classes'],
        dropout=model_config['dropout'],
        n_regions=model_config['n_regions']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n✓ Model loaded successfully")
        print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    return model, config, checkpoint


def load_data(h5_path, metadata_path, verbose=False):
    """
    Load HDF5 data and metadata
    
    Args:
        h5_path: Path to HDF5 file
        metadata_path: Path to metadata CSV
        verbose: Print information
    
    Returns:
        h5_file: Open HDF5 file object
        metadata: Metadata DataFrame
    """
    if verbose:
        print(f"\nLoading data:")
        print(f"  HDF5: {h5_path}")
        print(f"  Metadata: {metadata_path}")
    
    # Load HDF5
    h5_file = h5py.File(h5_path, 'r')
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    
    if verbose:
        print(f"\n  ✓ Loaded {len(metadata)} samples")
        print(f"  ✓ Methylation shape: {h5_file['methylation'].shape}")
        print(f"  ✓ Sample names: {list(metadata['sample_name'].head(5))}...")
    
    return h5_file, metadata


def run_inference_batch(model, h5_file, metadata, device, batch_size=8, 
                       apply_renorm=False, renorm_strategy='threshold', 
                       renorm_threshold=0.05, verbose=False):
    """
    Run inference on all samples in batches
    
    Args:
        model: Trained model
        h5_file: HDF5 file with methylation data
        metadata: Metadata DataFrame
        device: Device for computation
        batch_size: Batch size for inference
        apply_renorm: Whether to apply renormalization
        renorm_strategy: Renormalization strategy
        renorm_threshold: Threshold for renormalization
        verbose: Print progress
    
    Returns:
        predictions: List of dicts with predictions for each sample
    """
    n_samples = len(metadata)
    predictions = []
    
    with torch.no_grad():
        # Process in batches
        for batch_start in tqdm(range(0, n_samples, batch_size), 
                               desc="Running inference", 
                               disable=not verbose):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_indices = range(batch_start, batch_end)
            
            # Load batch from HDF5
            batch_data = []
            for idx in batch_indices:
                methylation = h5_file['methylation'][idx]  # [n_regions, seq_length]
                batch_data.append(methylation)
            
            # Stack into batch tensor
            methylation_batch = torch.from_numpy(np.array(batch_data)).long().to(device)
            
            # Run inference
            if apply_renorm:
                proportions = model(
                    methylation_batch,
                    apply_renorm=True,
                    renorm_strategy=renorm_strategy,
                    renorm_params={'threshold': renorm_threshold}
                )
            else:
                proportions = model(methylation_batch, apply_renorm=False)
            
            # Convert to numpy and store results
            proportions_np = proportions.cpu().numpy()
            
            for i, idx in enumerate(batch_indices):
                row = metadata.iloc[idx]
                predictions.append({
                    'sample_idx': int(row['sample_idx']),
                    'sample_name': row['sample_name'],
                    'proportions': proportions_np[i],
                    'statistics': {
                        'methylation_pct': float(row.get('methylation_pct', 0)),
                        'missing_pct': float(row.get('missing_pct', 0))
                    }
                })
    
    return predictions


def create_prediction_dataframe(predictions, tissue_names):
    """
    Create DataFrame with all predictions
    
    Args:
        predictions: List of prediction dicts
        tissue_names: List of tissue names
    
    Returns:
        df: DataFrame with columns [sample_name, Tissue_00, Tissue_01, ...]
    """
    # Create data for DataFrame
    data = []
    for pred in predictions:
        row = {'sample_name': pred['sample_name']}
        for i, tissue_name in enumerate(tissue_names):
            row[tissue_name] = pred['proportions'][i]
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def generate_clinical_report(prediction, tissue_names, min_proportion=0.01):
    """
    Generate clinical report for a single sample
    
    Args:
        prediction: Prediction dict for one sample
        tissue_names: List of tissue names
        min_proportion: Minimum proportion to report
    
    Returns:
        report: String with formatted clinical report
    """
    sample_name = prediction['sample_name']
    proportions = prediction['proportions']
    stats = prediction['statistics']
    
    # Sort tissues by proportion (descending)
    tissue_props = [(tissue_names[i], proportions[i]) for i in range(len(tissue_names))]
    tissue_props.sort(key=lambda x: x[1], reverse=True)
    
    # Generate report
    report = []
    report.append("="*70)
    report.append(f"TISSUE DECONVOLUTION REPORT")
    report.append("="*70)
    report.append(f"Sample ID: {sample_name}")
    report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("Sample Quality Metrics:")
    report.append(f"  Methylation rate: {stats.get('methylation_pct', 0):.2f}%")
    report.append(f"  Missing data: {stats.get('missing_pct', 0):.2f}%")
    report.append("")
    report.append("Predicted Tissue Proportions:")
    report.append("-"*70)
    report.append(f"{'Tissue':<30} {'Proportion':<15} {'Percentage'}")
    report.append("-"*70)
    
    for tissue_name, proportion in tissue_props:
        if proportion >= min_proportion:
            report.append(f"{tissue_name:<30} {proportion:.4f}          {proportion*100:>6.2f}%")
    
    report.append("-"*70)
    report.append(f"{'TOTAL':<30} {proportions.sum():.4f}          {proportions.sum()*100:>6.2f}%")
    report.append("")
    
    # Clinical alerts (tissues >5% that might be elevated)
    elevated_tissues = [(name, prop) for name, prop in tissue_props 
                       if prop > 0.05 and 'Blood' not in name]
    
    if elevated_tissues:
        report.append("Clinical Alerts:")
        report.append("-"*70)
        for tissue_name, proportion in elevated_tissues:
            report.append(f"  ⚠️  Elevated {tissue_name}: {proportion*100:.2f}%")
            report.append(f"      → Consider enhanced surveillance for {tissue_name.lower()} metastasis")
        report.append("")
    
    report.append("="*70)
    
    return '\n'.join(report)


def save_results(predictions, tissue_names, output_dir, min_proportion, verbose=False):
    """
    Save all results to output directory
    
    Args:
        predictions: List of prediction dicts
        tissue_names: List of tissue names
        output_dir: Output directory
        min_proportion: Minimum proportion for clinical reports
        verbose: Print information
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\nSaving results to {output_dir}")
    
    # 1. Save predictions as CSV (all samples)
    df = create_prediction_dataframe(predictions, tissue_names)
    csv_path = output_dir / 'tissue_proportions_all_samples.csv'
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"  ✓ Saved predictions: {csv_path}")
    
    # 2. Save predictions as JSON (for programmatic access)
    json_data = {
        'tissue_names': tissue_names,
        'predictions': [
            {
                'sample_name': pred['sample_name'],
                'proportions': pred['proportions'].tolist(),
                'statistics': pred['statistics']
            }
            for pred in predictions
        ],
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(predictions),
            'n_tissues': len(tissue_names)
        }
    }
    json_path = output_dir / 'predictions.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    if verbose:
        print(f"  ✓ Saved JSON: {json_path}")
    
    # 3. Generate individual clinical reports
    reports_dir = output_dir / 'clinical_reports'
    reports_dir.mkdir(exist_ok=True)
    
    for pred in predictions:
        report = generate_clinical_report(pred, tissue_names, min_proportion)
        report_path = reports_dir / f"{pred['sample_name']}_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
    
    if verbose:
        print(f"  ✓ Saved {len(predictions)} clinical reports: {reports_dir}")
    
    # 4. Create summary statistics
    summary = {
        'n_samples': len(predictions),
        'n_tissues': len(tissue_names),
        'tissue_names': tissue_names,
        'per_tissue_stats': {}
    }
    
    # Calculate per-tissue statistics
    all_proportions = np.array([pred['proportions'] for pred in predictions])
    for i, tissue_name in enumerate(tissue_names):
        tissue_props = all_proportions[:, i]
        summary['per_tissue_stats'][tissue_name] = {
            'mean': float(tissue_props.mean()),
            'std': float(tissue_props.std()),
            'min': float(tissue_props.min()),
            'max': float(tissue_props.max()),
            'median': float(np.median(tissue_props))
        }
    
    summary_path = output_dir / 'summary_statistics.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"  ✓ Saved summary statistics: {summary_path}")
    
    # 5. Print summary to console
    if verbose:
        print(f"\n{'='*70}")
        print("INFERENCE SUMMARY")
        print(f"{'='*70}")
        print(f"Samples processed: {len(predictions)}")
        print(f"Tissues predicted: {len(tissue_names)}")
        print(f"\nTop 5 most abundant tissues (average across samples):")
        avg_props = all_proportions.mean(axis=0)
        top_tissues = sorted(zip(tissue_names, avg_props), key=lambda x: x[1], reverse=True)[:5]
        for tissue_name, prop in top_tissues:
            print(f"  {tissue_name:<30} {prop*100:>6.2f}%")
        print(f"{'='*70}")


def main():
    """Main inference pipeline"""
    args = parse_args()
    
    print("="*70)
    print("PDAC CLINICAL PREDICTION - INFERENCE PIPELINE")
    print("="*70)
    print(f"Input HDF5: {args.input_h5}")
    print(f"Input metadata: {args.input_metadata}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Renormalization: {args.apply_renormalization}")
    if args.apply_renormalization:
        print(f"  Strategy: {args.renorm_strategy}")
        print(f"  Threshold: {args.renorm_threshold}")
    print("="*70)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available, using CPU")
    
    # Load model
    model, config, checkpoint = load_checkpoint(
        args.checkpoint, 
        device=device, 
        verbose=args.verbose
    )
    
    # Load tissue names
    tissue_names = load_tissue_names(args.tissue_names, config['model']['num_classes'])
    if args.verbose:
        print(f"\nTissue names: {tissue_names}")
    
    # Load data
    h5_file, metadata = load_data(args.input_h5, args.input_metadata, verbose=args.verbose)
    
    # Run inference
    predictions = run_inference_batch(
        model=model,
        h5_file=h5_file,
        metadata=metadata,
        device=device,
        batch_size=args.batch_size,
        apply_renorm=args.apply_renormalization,
        renorm_strategy=args.renorm_strategy,
        renorm_threshold=args.renorm_threshold,
        verbose=args.verbose
    )
    
    # Close HDF5 file
    h5_file.close()
    
    # Save results
    save_results(
        predictions=predictions,
        tissue_names=tissue_names,
        output_dir=args.output_dir,
        min_proportion=args.min_proportion,
        verbose=args.verbose
    )
    
    print("\n✓ Inference completed successfully!")
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - Predictions: tissue_proportions_all_samples.csv")
    print(f"  - JSON: predictions.json")
    print(f"  - Clinical reports: clinical_reports/")
    print(f"  - Summary: summary_statistics.json")


if __name__ == '__main__':
    main()
