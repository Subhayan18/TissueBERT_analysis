#!/usr/bin/env python3
"""
Validate Generated Mixture Datasets
====================================

This script performs comprehensive validation of the generated mixture datasets,
checking data structures, properties, annotations, and numerical statistics.

Generates a detailed report for all 6 datasets (3 phases × 2 splits).

Author: Mixture Deconvolution Project
Date: December 2024
"""

import numpy as np
import pandas as pd
import h5py
import json
from pathlib import Path
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_mixture_dataset(filepath: str) -> Dict:
    """Load mixture dataset and extract all information."""
    print(f"\nLoading: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        data = {
            'mixed_methylation': f['mixed_methylation'][:],
            'true_proportions': f['true_proportions'][:],
            'mixture_info': [json.loads(s) for s in f['mixture_info'][:]],
            'tissue_names': [t.decode() if isinstance(t, bytes) else t 
                           for t in f.attrs['tissue_names']],
            'n_tissues': f.attrs['n_tissues'],
            'n_regions': f.attrs['n_regions'],
            'n_mixtures': f.attrs['n_mixtures'],
            'phase': f.attrs['phase'],
            'split': f.attrs['split']
        }
    
    print(f"  ✓ Loaded {data['n_mixtures']} mixtures")
    return data


def validate_data_structure(data: Dict) -> Dict:
    """Validate basic data structure and types."""
    print("\n  Validating data structure...")
    
    results = {}
    
    # Check array shapes
    mixed_shape = data['mixed_methylation'].shape
    props_shape = data['true_proportions'].shape
    info_len = len(data['mixture_info'])
    
    results['shapes_match'] = (
        mixed_shape[0] == props_shape[0] == info_len == data['n_mixtures']
    )
    results['mixed_shape'] = mixed_shape
    results['props_shape'] = props_shape
    results['info_length'] = info_len
    
    # Check dimensions
    results['n_regions_match'] = mixed_shape[1] == data['n_regions']
    results['n_tissues_match'] = props_shape[1] == data['n_tissues']
    
    # Check data types
    results['mixed_dtype'] = str(data['mixed_methylation'].dtype)
    results['props_dtype'] = str(data['true_proportions'].dtype)
    
    # Check for NaN/Inf
    results['mixed_has_nan'] = np.isnan(data['mixed_methylation']).any()
    results['mixed_has_inf'] = np.isinf(data['mixed_methylation']).any()
    results['props_has_nan'] = np.isnan(data['true_proportions']).any()
    results['props_has_inf'] = np.isinf(data['true_proportions']).any()
    
    # Value ranges
    results['mixed_min'] = float(np.nanmin(data['mixed_methylation']))
    results['mixed_max'] = float(np.nanmax(data['mixed_methylation']))
    results['props_min'] = float(np.nanmin(data['true_proportions']))
    results['props_max'] = float(np.nanmax(data['true_proportions']))
    
    # Proportion sum check
    prop_sums = data['true_proportions'].sum(axis=1)
    results['props_sum_min'] = float(prop_sums.min())
    results['props_sum_max'] = float(prop_sums.max())
    results['props_sum_mean'] = float(prop_sums.mean())
    results['props_all_sum_to_1'] = np.allclose(prop_sums, 1.0, atol=1e-5)
    
    print(f"    Shapes match: {results['shapes_match']}")
    print(f"    Proportions sum to 1.0: {results['props_all_sum_to_1']}")
    print(f"    No NaN/Inf: {not (results['mixed_has_nan'] or results['props_has_nan'])}")
    
    return results


def analyze_mixture_composition(data: Dict) -> Dict:
    """Analyze mixture composition and tissue statistics."""
    print("\n  Analyzing mixture composition...")
    
    results = {}
    phase = data['phase']
    
    # Number of tissues per mixture
    n_tissues_per_mixture = (data['true_proportions'] > 0).sum(axis=1)
    results['n_tissues_per_mixture'] = {
        'min': int(n_tissues_per_mixture.min()),
        'max': int(n_tissues_per_mixture.max()),
        'mean': float(n_tissues_per_mixture.mean()),
        'median': float(np.median(n_tissues_per_mixture)),
        'distribution': Counter(n_tissues_per_mixture.tolist())
    }
    
    # Tissue frequency (how often each tissue appears)
    tissue_frequency = (data['true_proportions'] > 0).sum(axis=0)
    results['tissue_frequency'] = {
        tissue: int(freq) 
        for tissue, freq in zip(data['tissue_names'], tissue_frequency)
    }
    results['tissue_freq_stats'] = {
        'min': int(tissue_frequency.min()),
        'max': int(tissue_frequency.max()),
        'mean': float(tissue_frequency.mean()),
        'std': float(tissue_frequency.std())
    }
    
    # Proportion statistics (for non-zero proportions)
    nonzero_props = data['true_proportions'][data['true_proportions'] > 0]
    results['proportion_stats'] = {
        'min': float(nonzero_props.min()),
        'max': float(nonzero_props.max()),
        'mean': float(nonzero_props.mean()),
        'median': float(np.median(nonzero_props)),
        'std': float(nonzero_props.std()),
        'q25': float(np.percentile(nonzero_props, 25)),
        'q75': float(np.percentile(nonzero_props, 75))
    }
    
    # Phase-specific analysis
    if phase == 1:
        # For 2-tissue mixtures, analyze proportion pairs
        prop_pairs = []
        for i in range(len(data['true_proportions'])):
            nonzero = data['true_proportions'][i][data['true_proportions'][i] > 0]
            if len(nonzero) == 2:
                prop_pairs.append((float(nonzero[0]), float(nonzero[1])))
        
        results['phase1_specific'] = {
            'n_pairs': len(prop_pairs),
            'unique_pairs': len(set(tuple(sorted(p)) for p in prop_pairs)),
            'pair_examples': prop_pairs[:10]
        }
    
    elif phase == 3:
        # For cfDNA mixtures, analyze Blood proportion
        blood_idx = data['tissue_names'].index('Blood') if 'Blood' in data['tissue_names'] else None
        
        if blood_idx is not None:
            blood_props = data['true_proportions'][:, blood_idx]
            blood_present = blood_props > 0
            
            results['phase3_specific'] = {
                'blood_present_count': int(blood_present.sum()),
                'blood_proportion_stats': {
                    'min': float(blood_props[blood_present].min()) if blood_present.any() else 0,
                    'max': float(blood_props[blood_present].max()) if blood_present.any() else 0,
                    'mean': float(blood_props[blood_present].mean()) if blood_present.any() else 0,
                    'median': float(np.median(blood_props[blood_present])) if blood_present.any() else 0,
                }
            }
    
    print(f"    Tissues per mixture: {results['n_tissues_per_mixture']['mean']:.1f} (range: {results['n_tissues_per_mixture']['min']}-{results['n_tissues_per_mixture']['max']})")
    print(f"    Proportion range: [{results['proportion_stats']['min']:.3f}, {results['proportion_stats']['max']:.3f}]")
    
    return results


def analyze_methylation_patterns(data: Dict) -> Dict:
    """Analyze methylation pattern statistics."""
    print("\n  Analyzing methylation patterns...")
    
    results = {}
    
    mixed_meth = data['mixed_methylation']
    
    # Overall statistics
    results['overall_stats'] = {
        'mean': float(np.nanmean(mixed_meth)),
        'median': float(np.nanmedian(mixed_meth)),
        'std': float(np.nanstd(mixed_meth)),
        'min': float(np.nanmin(mixed_meth)),
        'max': float(np.nanmax(mixed_meth)),
        'q25': float(np.nanpercentile(mixed_meth, 25)),
        'q75': float(np.nanpercentile(mixed_meth, 75))
    }
    
    # Per-sample statistics
    sample_means = np.nanmean(mixed_meth, axis=1)
    results['per_sample_mean_stats'] = {
        'min': float(sample_means.min()),
        'max': float(sample_means.max()),
        'mean': float(sample_means.mean()),
        'std': float(sample_means.std())
    }
    
    # Per-region statistics
    region_means = np.nanmean(mixed_meth, axis=0)
    results['per_region_mean_stats'] = {
        'min': float(region_means.min()),
        'max': float(region_means.max()),
        'mean': float(region_means.mean()),
        'std': float(region_means.std())
    }
    
    # Coverage (non-NaN values)
    coverage = (~np.isnan(mixed_meth)).sum() / mixed_meth.size
    results['coverage'] = float(coverage)
    
    # Methylation level distribution
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, _ = np.histogram(mixed_meth[~np.isnan(mixed_meth)], bins=bins)
    results['methylation_distribution'] = {
        '0.0-0.2': int(hist[0]),
        '0.2-0.4': int(hist[1]),
        '0.4-0.6': int(hist[2]),
        '0.6-0.8': int(hist[3]),
        '0.8-1.0': int(hist[4])
    }
    
    print(f"    Mean methylation: {results['overall_stats']['mean']:.3f}")
    print(f"    Coverage: {results['coverage']*100:.1f}%")
    
    return results


def analyze_metadata(data: Dict) -> Dict:
    """Analyze mixture metadata."""
    print("\n  Analyzing mixture metadata...")
    
    results = {}
    
    mixture_info = data['mixture_info']
    phase = data['phase']
    
    # Augmentation version distribution
    aug_versions = [info['aug_version'] for info in mixture_info]
    results['aug_version_distribution'] = dict(Counter(aug_versions))
    
    if phase == 1:
        # Phase 1 specific: tissue pairs and proportions
        tissue_pairs = [(info['tissue_a'], info['tissue_b']) for info in mixture_info]
        unique_pairs = set(tuple(sorted([p[0], p[1]])) for p in tissue_pairs)
        
        prop_combinations = [(info['prop_a'], info['prop_b']) for info in mixture_info]
        unique_prop_combos = set(tuple(sorted([p[0], p[1]])) for p in prop_combinations)
        
        results['phase1_metadata'] = {
            'total_pairs': len(tissue_pairs),
            'unique_tissue_pairs': len(unique_pairs),
            'unique_proportion_combos': len(unique_prop_combos),
            'most_common_pairs': Counter([tuple(sorted([p[0], p[1]])) for p in tissue_pairs]).most_common(10)
        }
        
    elif phase == 2:
        # Phase 2 specific: number of components
        n_components = [info['n_components'] for info in mixture_info]
        results['phase2_metadata'] = {
            'n_components_distribution': dict(Counter(n_components))
        }
        
    elif phase == 3:
        # Phase 3 specific: Blood proportion and other tissues
        blood_props = [info['blood_proportion'] for info in mixture_info]
        n_other = [info['n_other_tissues'] for info in mixture_info]
        
        results['phase3_metadata'] = {
            'blood_proportion_stats': {
                'min': float(np.min(blood_props)),
                'max': float(np.max(blood_props)),
                'mean': float(np.mean(blood_props)),
                'median': float(np.median(blood_props))
            },
            'n_other_tissues_distribution': dict(Counter(n_other))
        }
    
    print(f"    Augmentation versions: {len(results['aug_version_distribution'])} unique")
    
    return results


def check_critical_constraints(data: Dict) -> Dict:
    """Check critical constraints from the mixing strategy."""
    print("\n  Checking critical constraints...")
    
    results = {}
    issues = []
    
    # Constraint 1: Proportions sum to 1.0
    prop_sums = data['true_proportions'].sum(axis=1)
    props_valid = np.allclose(prop_sums, 1.0, atol=1e-5)
    results['proportions_sum_to_1'] = bool(props_valid)
    if not props_valid:
        issues.append(f"Proportions don't sum to 1.0 (range: {prop_sums.min():.6f} - {prop_sums.max():.6f})")
    
    # Constraint 2: All proportions non-negative
    all_non_negative = (data['true_proportions'] >= 0).all()
    results['proportions_non_negative'] = bool(all_non_negative)
    if not all_non_negative:
        issues.append("Some proportions are negative")
    
    # Constraint 3: Methylation in valid range [0, 1]
    mixed_meth = data['mixed_methylation']
    valid_mask = ~np.isnan(mixed_meth)
    in_range = ((mixed_meth[valid_mask] >= 0) & (mixed_meth[valid_mask] <= 1)).all()
    results['methylation_in_range'] = bool(in_range)
    if not in_range:
        issues.append(f"Methylation outside [0,1] range: [{mixed_meth[valid_mask].min():.3f}, {mixed_meth[valid_mask].max():.3f}]")
    
    # Constraint 4: No NaN/Inf in proportions
    no_invalid_props = not (np.isnan(data['true_proportions']).any() or np.isinf(data['true_proportions']).any())
    results['proportions_no_nan_inf'] = bool(no_invalid_props)
    if not no_invalid_props:
        issues.append("NaN or Inf found in proportions")
    
    # Constraint 5: Phase-specific checks
    phase = data['phase']
    n_tissues_per_mixture = (data['true_proportions'] > 0).sum(axis=1)
    
    if phase == 1:
        all_2_tissues = (n_tissues_per_mixture == 2).all()
        results['phase1_all_2_tissues'] = bool(all_2_tissues)
        if not all_2_tissues:
            issues.append(f"Phase 1: Not all mixtures have exactly 2 tissues (found: {set(n_tissues_per_mixture.tolist())})")
    
    elif phase == 2:
        in_range_3_5 = ((n_tissues_per_mixture >= 3) & (n_tissues_per_mixture <= 5)).all()
        results['phase2_3_to_5_tissues'] = bool(in_range_3_5)
        if not in_range_3_5:
            issues.append(f"Phase 2: Not all mixtures have 3-5 tissues (found: {set(n_tissues_per_mixture.tolist())})")
    
    elif phase == 3:
        # Check Blood presence
        if 'Blood' in data['tissue_names']:
            blood_idx = data['tissue_names'].index('Blood')
            blood_present = (data['true_proportions'][:, blood_idx] > 0).all()
            results['phase3_blood_always_present'] = bool(blood_present)
            if not blood_present:
                issues.append("Phase 3: Blood not present in all mixtures")
            
            # Check Blood dominance (60-90%)
            blood_props = data['true_proportions'][:, blood_idx]
            blood_dominant = ((blood_props >= 0.6) & (blood_props <= 0.9)).all()
            results['phase3_blood_dominant'] = bool(blood_dominant)
            if not blood_dominant:
                issues.append(f"Phase 3: Blood not in 60-90% range (found: {blood_props.min():.2f} - {blood_props.max():.2f})")
    
    results['issues'] = issues
    results['all_constraints_passed'] = len(issues) == 0
    
    if results['all_constraints_passed']:
        print(f"    ✓ All critical constraints passed")
    else:
        print(f"    ✗ Found {len(issues)} constraint violations")
        for issue in issues:
            print(f"      - {issue}")
    
    return results


def generate_visualizations(data: Dict, output_dir: Path, dataset_name: str):
    """Generate visualizations for the dataset."""
    print("\n  Generating visualizations...")
    
    fig_dir = output_dir / 'figures' / dataset_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Proportion distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of non-zero proportions
    ax = axes[0]
    nonzero_props = data['true_proportions'][data['true_proportions'] > 0]
    ax.hist(nonzero_props, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Proportion Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Non-Zero Proportions', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Number of tissues per mixture
    ax = axes[1]
    n_tissues = (data['true_proportions'] > 0).sum(axis=1)
    tissue_counts = Counter(n_tissues.tolist())
    ax.bar(tissue_counts.keys(), tissue_counts.values(), edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Tissues', fontsize=12)
    ax.set_ylabel('Number of Mixtures', fontsize=12)
    ax.set_title('Tissues per Mixture Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'proportion_distributions.png', bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: proportion_distributions.png")
    
    # Figure 2: Tissue frequency
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    tissue_freq = (data['true_proportions'] > 0).sum(axis=0)
    sorted_indices = np.argsort(tissue_freq)[::-1]
    
    ax.bar(range(len(tissue_freq)), tissue_freq[sorted_indices], edgecolor='black', alpha=0.7)
    ax.set_xticks(range(len(tissue_freq)))
    ax.set_xticklabels([data['tissue_names'][i] for i in sorted_indices], 
                       rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Tissue', fontsize=12)
    ax.set_ylabel('Frequency (number of mixtures)', fontsize=12)
    ax.set_title('Tissue Frequency Across All Mixtures', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'tissue_frequency.png', bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: tissue_frequency.png")
    
    # Figure 3: Methylation distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall methylation histogram
    ax = axes[0]
    valid_meth = data['mixed_methylation'][~np.isnan(data['mixed_methylation'])]
    ax.hist(valid_meth, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Methylation Level', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Overall Methylation Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Per-sample mean methylation
    ax = axes[1]
    sample_means = np.nanmean(data['mixed_methylation'], axis=1)
    ax.hist(sample_means, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel('Mean Methylation per Sample', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Per-Sample Mean Methylation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'methylation_distributions.png', bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: methylation_distributions.png")
    
    # Phase-specific visualizations
    if data['phase'] == 3 and 'Blood' in data['tissue_names']:
        # Blood proportion distribution
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        blood_idx = data['tissue_names'].index('Blood')
        blood_props = data['true_proportions'][:, blood_idx]
        blood_props = blood_props[blood_props > 0]
        
        ax.hist(blood_props, bins=30, edgecolor='black', alpha=0.7, color='red')
        ax.axvline(x=0.6, color='blue', linestyle='--', linewidth=2, label='Min (60%)')
        ax.axvline(x=0.9, color='blue', linestyle='--', linewidth=2, label='Max (90%)')
        ax.set_xlabel('Blood Proportion', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Phase 3: Blood Proportion Distribution\n(Should be 60-90%)', 
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'blood_proportion_phase3.png', bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: blood_proportion_phase3.png")


def generate_dataset_report(data: Dict, output_dir: Path, dataset_name: str):
    """Generate comprehensive report for a single dataset."""
    print(f"\n{'='*80}")
    print(f"VALIDATING: {dataset_name}")
    print(f"{'='*80}")
    
    # Run all validation checks
    structure_results = validate_data_structure(data)
    composition_results = analyze_mixture_composition(data)
    methylation_results = analyze_methylation_patterns(data)
    metadata_results = analyze_metadata(data)
    constraint_results = check_critical_constraints(data)
    
    # Generate visualizations
    generate_visualizations(data, output_dir, dataset_name)
    
    # Compile report
    report = []
    report.append("="*80)
    report.append(f"DATASET VALIDATION REPORT: {dataset_name}")
    report.append("="*80)
    report.append("")
    
    # Basic Info
    report.append("1. BASIC INFORMATION")
    report.append("-"*80)
    report.append(f"Phase: {data['phase']}")
    report.append(f"Split: {data['split']}")
    report.append(f"Number of mixtures: {data['n_mixtures']}")
    report.append(f"Number of tissues: {data['n_tissues']}")
    report.append(f"Number of regions: {data['n_regions']}")
    report.append("")
    
    # Data Structure
    report.append("2. DATA STRUCTURE VALIDATION")
    report.append("-"*80)
    report.append(f"Mixed methylation shape: {structure_results['mixed_shape']}")
    report.append(f"True proportions shape: {structure_results['props_shape']}")
    report.append(f"Mixture info length: {structure_results['info_length']}")
    report.append(f"Shapes match: {structure_results['shapes_match']}")
    report.append(f"Data types: mixed={structure_results['mixed_dtype']}, props={structure_results['props_dtype']}")
    report.append("")
    report.append(f"Data quality:")
    report.append(f"  Mixed methylation - NaN: {structure_results['mixed_has_nan']}, Inf: {structure_results['mixed_has_inf']}")
    report.append(f"  True proportions - NaN: {structure_results['props_has_nan']}, Inf: {structure_results['props_has_inf']}")
    report.append(f"  Value ranges:")
    report.append(f"    Mixed methylation: [{structure_results['mixed_min']:.6f}, {structure_results['mixed_max']:.6f}]")
    report.append(f"    True proportions: [{structure_results['props_min']:.6f}, {structure_results['props_max']:.6f}]")
    report.append("")
    report.append(f"Proportion sums:")
    report.append(f"  Min: {structure_results['props_sum_min']:.6f}")
    report.append(f"  Max: {structure_results['props_sum_max']:.6f}")
    report.append(f"  Mean: {structure_results['props_sum_mean']:.6f}")
    report.append(f"  All sum to 1.0: {structure_results['props_all_sum_to_1']}")
    report.append("")
    
    # Mixture Composition
    report.append("3. MIXTURE COMPOSITION ANALYSIS")
    report.append("-"*80)
    report.append(f"Tissues per mixture:")
    report.append(f"  Min: {composition_results['n_tissues_per_mixture']['min']}")
    report.append(f"  Max: {composition_results['n_tissues_per_mixture']['max']}")
    report.append(f"  Mean: {composition_results['n_tissues_per_mixture']['mean']:.2f}")
    report.append(f"  Median: {composition_results['n_tissues_per_mixture']['median']:.2f}")
    report.append(f"  Distribution: {dict(composition_results['n_tissues_per_mixture']['distribution'])}")
    report.append("")
    report.append(f"Tissue frequency statistics:")
    report.append(f"  Min: {composition_results['tissue_freq_stats']['min']}")
    report.append(f"  Max: {composition_results['tissue_freq_stats']['max']}")
    report.append(f"  Mean: {composition_results['tissue_freq_stats']['mean']:.1f}")
    report.append(f"  Std: {composition_results['tissue_freq_stats']['std']:.1f}")
    report.append("")
    report.append(f"Proportion statistics (non-zero only):")
    report.append(f"  Min: {composition_results['proportion_stats']['min']:.6f}")
    report.append(f"  Max: {composition_results['proportion_stats']['max']:.6f}")
    report.append(f"  Mean: {composition_results['proportion_stats']['mean']:.6f}")
    report.append(f"  Median: {composition_results['proportion_stats']['median']:.6f}")
    report.append(f"  Std: {composition_results['proportion_stats']['std']:.6f}")
    report.append(f"  Q25: {composition_results['proportion_stats']['q25']:.6f}")
    report.append(f"  Q75: {composition_results['proportion_stats']['q75']:.6f}")
    report.append("")
    
    # Phase-specific
    if 'phase1_specific' in composition_results:
        report.append(f"Phase 1 specific analysis:")
        report.append(f"  Number of 2-tissue pairs: {composition_results['phase1_specific']['n_pairs']}")
        report.append(f"  Unique proportion combinations: {composition_results['phase1_specific']['unique_pairs']}")
        report.append("")
    elif 'phase3_specific' in composition_results:
        report.append(f"Phase 3 specific analysis:")
        report.append(f"  Blood present in {composition_results['phase3_specific']['blood_present_count']} mixtures")
        report.append(f"  Blood proportion statistics:")
        for key, val in composition_results['phase3_specific']['blood_proportion_stats'].items():
            report.append(f"    {key}: {val:.6f}")
        report.append("")
    
    # Top 10 most frequent tissues
    report.append(f"Top 10 most frequent tissues:")
    sorted_tissues = sorted(composition_results['tissue_frequency'].items(), 
                          key=lambda x: x[1], reverse=True)
    for tissue, freq in sorted_tissues[:10]:
        report.append(f"  {tissue}: {freq}")
    report.append("")
    
    # Methylation Patterns
    report.append("4. METHYLATION PATTERN ANALYSIS")
    report.append("-"*80)
    report.append(f"Overall statistics:")
    for key, val in methylation_results['overall_stats'].items():
        report.append(f"  {key}: {val:.6f}")
    report.append("")
    report.append(f"Per-sample mean statistics:")
    for key, val in methylation_results['per_sample_mean_stats'].items():
        report.append(f"  {key}: {val:.6f}")
    report.append("")
    report.append(f"Coverage (non-NaN): {methylation_results['coverage']*100:.2f}%")
    report.append("")
    report.append(f"Methylation level distribution:")
    for range_name, count in methylation_results['methylation_distribution'].items():
        report.append(f"  {range_name}: {count}")
    report.append("")
    
    # Metadata
    report.append("5. METADATA ANALYSIS")
    report.append("-"*80)
    report.append(f"Augmentation version distribution:")
    for aug, count in sorted(metadata_results['aug_version_distribution'].items()):
        report.append(f"  aug{aug}: {count}")
    report.append("")
    
    if 'phase1_metadata' in metadata_results:
        report.append(f"Phase 1 metadata:")
        report.append(f"  Total tissue pairs: {metadata_results['phase1_metadata']['total_pairs']}")
        report.append(f"  Unique tissue pairs: {metadata_results['phase1_metadata']['unique_tissue_pairs']}")
        report.append(f"  Unique proportion combinations: {metadata_results['phase1_metadata']['unique_proportion_combos']}")
        report.append(f"  Most common tissue pairs:")
        for pair, count in metadata_results['phase1_metadata']['most_common_pairs']:
            report.append(f"    {pair}: {count}")
        report.append("")
    elif 'phase2_metadata' in metadata_results:
        report.append(f"Phase 2 metadata:")
        report.append(f"  Component distribution:")
        for n, count in sorted(metadata_results['phase2_metadata']['n_components_distribution'].items()):
            report.append(f"    {n} tissues: {count}")
        report.append("")
    elif 'phase3_metadata' in metadata_results:
        report.append(f"Phase 3 metadata:")
        report.append(f"  Blood proportion statistics:")
        for key, val in metadata_results['phase3_metadata']['blood_proportion_stats'].items():
            report.append(f"    {key}: {val:.6f}")
        report.append(f"  Number of other tissues distribution:")
        for n, count in sorted(metadata_results['phase3_metadata']['n_other_tissues_distribution'].items()):
            report.append(f"    {n} other tissues: {count}")
        report.append("")
    
    # Critical Constraints
    report.append("6. CRITICAL CONSTRAINT VALIDATION")
    report.append("-"*80)
    report.append(f"All constraints passed: {constraint_results['all_constraints_passed']}")
    report.append("")
    report.append(f"Individual checks:")
    report.append(f"  Proportions sum to 1.0: {constraint_results['proportions_sum_to_1']}")
    report.append(f"  Proportions non-negative: {constraint_results['proportions_non_negative']}")
    report.append(f"  Methylation in [0,1] range: {constraint_results['methylation_in_range']}")
    report.append(f"  No NaN/Inf in proportions: {constraint_results['proportions_no_nan_inf']}")
    
    if data['phase'] == 1 and 'phase1_all_2_tissues' in constraint_results:
        report.append(f"  Phase 1: All mixtures have 2 tissues: {constraint_results['phase1_all_2_tissues']}")
    elif data['phase'] == 2 and 'phase2_3_to_5_tissues' in constraint_results:
        report.append(f"  Phase 2: All mixtures have 3-5 tissues: {constraint_results['phase2_3_to_5_tissues']}")
    elif data['phase'] == 3:
        if 'phase3_blood_always_present' in constraint_results:
            report.append(f"  Phase 3: Blood always present: {constraint_results['phase3_blood_always_present']}")
        if 'phase3_blood_dominant' in constraint_results:
            report.append(f"  Phase 3: Blood in 60-90% range: {constraint_results['phase3_blood_dominant']}")
    
    if constraint_results['issues']:
        report.append("")
        report.append(f"Issues found:")
        for issue in constraint_results['issues']:
            report.append(f"  ✗ {issue}")
    else:
        report.append("")
        report.append(f"✓ No issues found - all constraints satisfied")
    
    report.append("")
    report.append("="*80)
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Validate generated mixture datasets')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing mixture dataset HDF5 files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for validation reports')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("MIXTURE DATASET VALIDATION")
    print("="*80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Find all HDF5 files
    hdf5_files = sorted(input_dir.glob('*.h5'))
    
    if not hdf5_files:
        print(f"ERROR: No HDF5 files found in {input_dir}")
        return
    
    print(f"Found {len(hdf5_files)} dataset files:")
    for f in hdf5_files:
        print(f"  - {f.name}")
    print()
    
    # Process each dataset
    all_reports = []
    
    for hdf5_file in hdf5_files:
        dataset_name = hdf5_file.stem
        
        # Load dataset
        data = load_mixture_dataset(str(hdf5_file))
        
        # Generate report
        report = generate_dataset_report(data, output_dir, dataset_name)
        
        # Save individual report
        report_path = output_dir / f'{dataset_name}_validation_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        print(f"\n✓ Saved individual report: {report_path.name}")
        
        all_reports.append({
            'name': dataset_name,
            'lines': report
        })
    
    # Create combined report
    print(f"\n{'='*80}")
    print("GENERATING COMBINED REPORT")
    print(f"{'='*80}")
    
    combined_report = []
    combined_report.append("="*80)
    combined_report.append("MIXTURE DATASETS VALIDATION - COMBINED REPORT")
    combined_report.append("="*80)
    combined_report.append("")
    combined_report.append(f"Total datasets validated: {len(all_reports)}")
    combined_report.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    combined_report.append("")
    combined_report.append("="*80)
    combined_report.append("")
    
    for report_data in all_reports:
        combined_report.extend(report_data['lines'])
        combined_report.append("")
        combined_report.append("")
    
    # Save combined report
    combined_path = output_dir / 'ALL_DATASETS_VALIDATION_REPORT.txt'
    with open(combined_path, 'w') as f:
        f.write('\n'.join(combined_report))
    
    print(f"✓ Saved combined report: {combined_path.name}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("VALIDATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - ALL_DATASETS_VALIDATION_REPORT.txt (combined)")
    for report_data in all_reports:
        print(f"  - {report_data['name']}_validation_report.txt")
    print(f"\nFigures saved to: {output_dir / 'figures'}")
    print()


if __name__ == '__main__':
    main()
