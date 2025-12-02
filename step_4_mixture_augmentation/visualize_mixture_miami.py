#!/usr/bin/env python3
"""
Miami Plot Visualization for Mixture Deconvolution Results

Creates a mirrored (top/bottom) visualization showing true vs predicted tissue proportions
across all test samples, with tissues ordered alphabetically on x-axis.

Usage:
    python visualize_mixture_miami.py \\
        --checkpoint /path/to/checkpoint_best.pt \\
        --test_h5 /path/to/test_mixtures.h5 \\
        --output /path/to/output_miami.png \\
        --summary /path/to/summary_stats.csv \\
        --colors "#FF0000,#FF7F00,#FFFF00,#00FF00,#0000FF,#4B0082,#9400D3"
"""

import argparse
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def load_model_and_predict(checkpoint_path, test_h5_path, device='cuda'):
    """
    Load model from checkpoint and generate predictions on test set.
    
    Args:
        checkpoint_path: Path to model checkpoint
        test_h5_path: Path to test HDF5 file
        device: Device to run inference on
        
    Returns:
        true_props: [n_samples, n_tissues] array of true proportions
        pred_props: [n_samples, n_tissues] array of predicted proportions
        tissue_names: List of tissue names
        sample_ids: List of sample identifiers
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Import model architecture
    try:
        from model_deconvolution import TissueBERTDeconvolution
    except ImportError:
        print("ERROR: Cannot import TissueBERTDeconvolution from model_deconvolution.py")
        print("Make sure model_deconvolution.py is in the same directory")
        sys.exit(1)
    
    # Get model configuration from checkpoint
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    
    # Extract model parameters
    n_regions = model_config.get('n_regions', 51089)
    hidden_size = model_config.get('hidden_size', 512)
    num_classes = model_config.get('num_classes', 22)
    dropout = model_config.get('dropout', 0.1)
    intermediate_size = model_config.get('intermediate_size', 2048)
    
    print(f"Model configuration:")
    print(f"  n_regions: {n_regions}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_classes (tissues): {num_classes}")
    print(f"  intermediate_size: {intermediate_size}")
    print(f"  dropout: {dropout}")
    
    # Load tissue names from metadata CSV
    import pandas as pd
    metadata_csv = config.get('data', {}).get('metadata_csv', 
                   '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/combined_metadata.csv')
    
    print(f"\nLoading tissue names from: {metadata_csv}")
    metadata = pd.read_csv(metadata_csv)
    
    # Get unique tissue_top_level names (sorted alphabetically)
    unique_tissues = sorted(metadata['tissue_top_level'].unique())
    print(f"Found {len(unique_tissues)} unique tissues (tissue_top_level)")
    
    if len(unique_tissues) != num_classes:
        print(f"WARNING: Found {len(unique_tissues)} tissues but model expects {num_classes}")
        # Pad or truncate as needed
        if len(unique_tissues) < num_classes:
            unique_tissues.extend([f"Tissue_{i}" for i in range(len(unique_tissues), num_classes)])
        else:
            unique_tissues = unique_tissues[:num_classes]
    
    tissue_names = unique_tissues
    print(f"Using tissues: {tissue_names}")
    
    # Initialize model with correct parameters
    model = TissueBERTDeconvolution(
        n_regions=n_regions,
        hidden_size=hidden_size,
        num_classes=num_classes,
        dropout=dropout,
        intermediate_size=intermediate_size
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loading test data from {test_h5_path}...")
    with h5py.File(test_h5_path, 'r') as f:
        print(f"Available keys in H5 file: {list(f.keys())}")
        
        # Load mixed methylation data
        # Shape: [n_samples, n_regions, seq_len]
        methylation = torch.tensor(f['mixed_methylation'][:], dtype=torch.float32)
        true_props = f['true_proportions'][:]  # [n_samples, n_tissues]
        
        print(f"Loaded mixed_methylation shape: {methylation.shape}")
        print(f"Loaded true_proportions shape: {true_props.shape}")
        
        # Sample IDs - just use indices
        sample_ids = [f"Mixture_{i}" for i in range(len(methylation))]
    
    n_samples = len(methylation)
    num_tissues = num_classes  # Use consistent naming
    print(f"Running inference on {n_samples} samples...")
    print(f"Methylation input shape: {methylation.shape}")
    
    # Ensure methylation has correct shape [n_samples, n_regions, seq_len]
    if len(methylation.shape) == 2:
        # If shape is [n_samples, n_regions], add seq_len dimension
        print("WARNING: Methylation is 2D, expanding to 3D...")
        methylation = methylation.unsqueeze(-1)
    
    print(f"Final methylation shape for inference: {methylation.shape}")
    
    # Run inference in batches
    batch_size = 32
    pred_props = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_meth = methylation[i:i+batch_size].to(device)
            
            # Model only takes methylation input
            batch_pred = model(batch_meth)
            pred_props.append(batch_pred.cpu().numpy())
    
    pred_props = np.vstack(pred_props)
    
    print(f"Predictions complete: {pred_props.shape}")
    return true_props, pred_props, tissue_names, sample_ids


def generate_rainbow_colors(n_colors):
    """
    Generate rainbow colors for tissues.
    
    Args:
        n_colors: Number of colors to generate
        
    Returns:
        List of hex color codes
    """
    import matplotlib.pyplot as plt
    cmap = plt.colormaps['rainbow']  # Fixed deprecation
    colors = [cmap(i / n_colors) for i in range(n_colors)]
    # Convert to hex
    hex_colors = ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) 
                  for r, g, b, a in colors]
    return hex_colors


def plot_miami_mixture(true_props, pred_props, tissue_names, 
                       colors=None, min_proportion=0.01, 
                       output_path='miami_plot.png', figsize=(20, 10)):
    """
    Create Miami-style plot comparing true vs predicted tissue proportions.
    
    Args:
        true_props: [n_samples, n_tissues] array of true proportions
        pred_props: [n_samples, n_tissues] array of predicted proportions
        tissue_names: List of tissue names
        colors: List of hex colors (one per tissue) or None for rainbow
        min_proportion: Minimum proportion threshold (filter below this)
        output_path: Path to save the figure
        figsize: Figure size (width, height)
    """
    n_samples, n_tissues = true_props.shape
    
    print(f"\nGenerating Miami plot for {n_samples} samples, {n_tissues} tissues...")
    print(f"Filtering proportions < {min_proportion*100}%")
    
    # Generate colors if not provided
    if colors is None:
        colors = generate_rainbow_colors(n_tissues)
    elif len(colors) < n_tissues:
        # If not enough colors provided, cycle through them
        colors = colors * (n_tissues // len(colors) + 1)
        colors = colors[:n_tissues]
    
    # Sort tissues alphabetically
    tissue_order = np.argsort(tissue_names)
    tissue_names_sorted = [tissue_names[i] for i in tissue_order]
    true_props_sorted = true_props[:, tissue_order]
    pred_props_sorted = pred_props[:, tissue_order]
    colors_sorted = [colors[i] for i in tissue_order]
    
    # Create figure with two subplots (top and bottom)
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=figsize, 
                                             sharex=True, 
                                             gridspec_kw={'hspace': 0.02},
                                             constrained_layout=False)  # Disable to avoid warning
    
    # Calculate positions for each tissue cluster
    tissue_positions = []
    tissue_centers = []
    current_pos = 0
    gap_between_tissues = 0.0  # NO gap between tissues
    
    for tissue_idx in range(n_tissues):
        # Get all samples for this tissue
        true_vals = true_props_sorted[:, tissue_idx]
        pred_vals = pred_props_sorted[:, tissue_idx]
        
        # Filter by minimum proportion in TRUE values only (key change #1)
        keep_mask = (true_vals >= min_proportion)
        true_vals_filtered = true_vals[keep_mask]
        pred_vals_filtered = pred_vals[keep_mask]
        n_bars = len(true_vals_filtered)
        
        if n_bars > 0:
            # Sort by true proportions (descending)
            sort_idx = np.argsort(true_vals_filtered)[::-1]
            true_vals_sorted_desc = true_vals_filtered[sort_idx]
            pred_vals_sorted_desc = pred_vals_filtered[sort_idx]
            
            # Calculate bar positions (very thin bars, no gap within tissue)
            bar_width = 0.8 / max(n_bars, 1)  # Total width of 0.8 per tissue
            positions = current_pos + np.arange(n_bars) * bar_width
            
            # Get color with transparency
            color_rgba = to_rgba(colors_sorted[tissue_idx], alpha=0.7)
            
            # Plot bars in TOP panel (pointing down, negative values)
            ax_top.bar(positions, -true_vals_sorted_desc * 100,  # Negative for downward
                      width=bar_width, color=color_rgba, 
                      edgecolor='none', align='edge')
            
            # Plot bars in BOTTOM panel (pointing UP but INVERTED Y-AXIS for mirror effect)
            ax_bottom.bar(positions, pred_vals_sorted_desc * 100,  # POSITIVE values
                         width=bar_width, color=color_rgba,
                         edgecolor='none', align='edge')
            
            tissue_positions.append((current_pos, current_pos + n_bars * bar_width))
            tissue_centers.append(current_pos + (n_bars * bar_width) / 2)
            current_pos += n_bars * bar_width + gap_between_tissues
        else:
            # Empty tissue - still add to maintain spacing
            tissue_positions.append((current_pos, current_pos + 0.8))
            tissue_centers.append(current_pos + 0.4)
            current_pos += 0.8 + gap_between_tissues
    
    # Set y-axis limits (0-100%)
    ax_top.set_ylim(-100, 0)
    ax_bottom.set_ylim(0, 100)  # POSITIVE for bottom (will be inverted)
    
    # Set y-axis labels
    ax_top.set_ylabel('True Proportion (%)', fontsize=12, fontweight='bold')
    ax_bottom.set_ylabel('Predicted Proportion (%)', fontsize=12, fontweight='bold')
    
    # Invert axes for mirror effect
    ax_top.invert_yaxis()  # Top: 0 at top, -100 at bottom
    ax_bottom.invert_yaxis()  # Bottom: 100 at top, 0 at bottom (MIRROR!)
    
    # Remove y-axis tick labels (keep only titles)
    ax_top.set_yticklabels([])
    ax_bottom.set_yticklabels([])
    
    # Remove y-axis ticks entirely
    ax_top.tick_params(axis='y', which='both', length=0)
    ax_bottom.tick_params(axis='y', which='both', length=0)
    
    # Set x-axis properties (key change #5 & #7)
    ax_bottom.set_xlim(-0.5, current_pos)
    ax_bottom.set_xticks(tissue_centers)
    ax_bottom.set_xticklabels(tissue_names_sorted, rotation=90, ha='center', 
                              fontsize=9)
    ax_bottom.set_xlabel('')
    
    # Remove x-axis ticks (key change #7)
    ax_bottom.tick_params(axis='x', which='both', length=0)
    ax_top.tick_params(axis='x', which='both', length=0)
    
    # Add grid
    ax_top.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax_bottom.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Add horizontal line at y=0 for both panels
    ax_top.axhline(y=0, color='black', linewidth=1.5, linestyle='-')
    ax_bottom.axhline(y=0, color='black', linewidth=1.5, linestyle='-')
    
    # Remove all spines (borders) - key change #3
    for ax in [ax_top, ax_bottom]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    
    # Add title
    fig.suptitle(f'Mixture Deconvolution: True vs Predicted Tissue Proportions\n'
                 f'{n_samples} samples, {n_tissues} tissues', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # NO LEGEND (key change #6)
    
    # Adjust layout manually to avoid tight_layout warning
    plt.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.15, hspace=0.02)
    
    # Save figure
    print(f"Saving Miami plot to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Miami plot saved successfully!")


def generate_summary_stats(true_props, pred_props, tissue_names, sample_ids, 
                          output_path='summary_stats.csv'):
    """
    Generate summary statistics CSV with per-sample per-tissue proportions.
    
    Args:
        true_props: [n_samples, n_tissues] array
        pred_props: [n_samples, n_tissues] array
        tissue_names: List of tissue names
        sample_ids: List of sample identifiers
        output_path: Path to save CSV file
    """
    print(f"\nGenerating summary statistics...")
    
    n_samples, n_tissues = true_props.shape
    
    # Create list of dictionaries for DataFrame
    records = []
    
    for sample_idx in range(n_samples):
        for tissue_idx in range(n_tissues):
            records.append({
                'sample_id': sample_ids[sample_idx],
                'tissue': tissue_names[tissue_idx],
                'true_proportion': true_props[sample_idx, tissue_idx],
                'predicted_proportion': pred_props[sample_idx, tissue_idx],
                'absolute_error': abs(true_props[sample_idx, tissue_idx] - 
                                    pred_props[sample_idx, tissue_idx])
            })
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Sort by sample_id then tissue
    df = df.sort_values(['sample_id', 'tissue'])
    
    # Save to CSV
    print(f"Saving summary statistics to {output_path}...")
    df.to_csv(output_path, index=False, float_format='%.6f')
    
    # Print summary
    print(f"\nSummary Statistics:")
    print(f"  Total records: {len(df)}")
    print(f"  Samples: {n_samples}")
    print(f"  Tissues: {n_tissues}")
    print(f"  Mean Absolute Error: {df['absolute_error'].mean():.4f}")
    print(f"  Median Absolute Error: {df['absolute_error'].median():.4f}")
    print(f"\nSummary statistics saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Generate Miami plot for mixture deconvolution results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default rainbow colors
    python visualize_mixture_miami.py \\
        --checkpoint /path/to/checkpoint_best.pt \\
        --test_h5 /path/to/phase2_test_mixtures.h5 \\
        --output miami_plot.png \\
        --summary summary_stats.csv
    
    # With custom colors (comma-separated hex codes)
    python visualize_mixture_miami.py \\
        --checkpoint checkpoint_best.pt \\
        --test_h5 test_mixtures.h5 \\
        --colors "#FF0000,#00FF00,#0000FF,#FFFF00,#FF00FF" \\
        --min_proportion 0.02
        """
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--test_h5', type=str, required=True,
                       help='Path to test HDF5 file')
    parser.add_argument('--output', type=str, default='miami_plot.png',
                       help='Output path for Miami plot (default: miami_plot.png)')
    parser.add_argument('--summary', type=str, default='summary_stats.csv',
                       help='Output path for summary statistics CSV (default: summary_stats.csv)')
    parser.add_argument('--colors', type=str, default=None,
                       help='Comma-separated hex color codes (e.g., "#FF0000,#00FF00,#0000FF")')
    parser.add_argument('--min_proportion', type=float, default=0.01,
                       help='Minimum proportion threshold (default: 0.01 = 1%%)')
    parser.add_argument('--figsize', type=str, default='20,10',
                       help='Figure size as "width,height" (default: "20,10")')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for inference: cuda or cpu (default: cuda)')
    
    args = parser.parse_args()
    
    # Parse colors if provided
    colors = None
    if args.colors:
        colors = [c.strip() for c in args.colors.split(',')]
        print(f"Using custom colors: {colors}")
    
    # Parse figsize
    figsize = tuple(map(float, args.figsize.split(',')))
    
    # Load model and generate predictions
    true_props, pred_props, tissue_names, sample_ids = load_model_and_predict(
        args.checkpoint, args.test_h5, device=args.device
    )
    
    # Generate Miami plot
    plot_miami_mixture(
        true_props=true_props,
        pred_props=pred_props,
        tissue_names=tissue_names,
        colors=colors,
        min_proportion=args.min_proportion,
        output_path=args.output,
        figsize=figsize
    )
    
    # Generate summary statistics
    generate_summary_stats(
        true_props=true_props,
        pred_props=pred_props,
        tissue_names=tissue_names,
        sample_ids=sample_ids,
        output_path=args.summary
    )
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"Miami plot: {args.output}")
    print(f"Summary stats: {args.summary}")
    print("="*60)


if __name__ == '__main__':
    main()
