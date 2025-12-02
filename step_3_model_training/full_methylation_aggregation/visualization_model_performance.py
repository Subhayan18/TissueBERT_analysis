#!/usr/bin/env python3
"""
Comprehensive Visualization: Single-Tissue Model Performance
=============================================================

Generates extensive visualizations for the trained single-tissue classification model:
1. UMAP embeddings (colored by tissue type)
2. Confusion matrices (validation + test)
3. Per-tissue accuracy heatmaps
4. Prediction confidence distributions
5. Feature space analysis
6. Misclassification analysis
7. Augmentation consistency analysis

Uses validation + test sets with all augmentation versions.

Author: TissueBERT Analysis
Date: November 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
import umap
from collections import defaultdict
import os
import sys

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Paths
HDF5_PATH = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/methylation_dataset.h5'
METADATA_PATH = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/combined_metadata.csv'
CHECKPOINT_PATH = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/fullgenome_results/checkpoints/checkpoint_best_acc.pt'
VAL_SPLIT_PATH = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/val_files.csv'
TEST_SPLIT_PATH = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/test_files.csv'
OUTPUT_DIR = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/single_tissue_visualizations'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)

print("="*80)
print("SINGLE-TISSUE MODEL VISUALIZATION")
print("Comprehensive Performance Analysis on Validation + Test Sets")
print("="*80)

# ============================================================================
# STEP 1: Load Model
# ============================================================================

print("\nSTEP 1: Loading trained model...")

class TissueBERT(nn.Module):
    def __init__(self, n_regions=51089, hidden_size=512, num_classes=22, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(n_regions, hidden_size)
        self.network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),      # network.0
            nn.BatchNorm1d(hidden_size),              # network.1
            nn.ReLU(),                                 # network.2
            nn.Dropout(dropout),                       # network.3
            nn.Linear(hidden_size, hidden_size),      # network.4
            nn.BatchNorm1d(hidden_size),              # network.5
            nn.ReLU(),                                 # network.6
            nn.Dropout(dropout),                       # network.7
            nn.Linear(hidden_size, 2048),             # network.8
            nn.BatchNorm1d(2048),                     # network.9
            nn.ReLU(),                                 # network.10
            nn.Dropout(dropout),                       # network.11
            nn.Linear(2048, num_classes)              # network.12
        )
    
    def forward(self, region_means):
        features = self.input_projection(region_means)
        logits = self.network(features)
        return logits
    
    def get_features(self, region_means):
        """Extract intermediate features for visualization"""
        features = self.input_projection(region_means)
        # Get features after second hidden layer (before final expansion)
        for i in range(8):  # Up to network.7 (before network.8)
            features = self.network[i](features)
        return features

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = TissueBERT(n_regions=51089, hidden_size=512, num_classes=22, dropout=0.1)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"✓ Loaded model from: {CHECKPOINT_PATH}")
print(f"  Training accuracy: {checkpoint.get('best_val_acc', 'N/A')}")

# ============================================================================
# STEP 2: Load Data
# ============================================================================

print("\nSTEP 2: Loading data and splits...")

# Load metadata
metadata = pd.read_csv(METADATA_PATH)
print(f"✓ Loaded metadata: {len(metadata)} samples")

# Load validation and test splits
val_files = pd.read_csv(VAL_SPLIT_PATH)
test_files = pd.read_csv(TEST_SPLIT_PATH)

print(f"✓ Validation set: {len(val_files)} files")
print(f"✓ Test set: {len(test_files)} files")

# Combine val + test
eval_files = pd.concat([val_files, test_files], ignore_index=True)
print(f"✓ Total evaluation files: {len(eval_files)}")

# Map filenames to indices
filename_to_idx = {row['filename']: idx for idx, row in metadata.iterrows()}

eval_indices = []
eval_split_labels = []  # 'val' or 'test'

for _, row in val_files.iterrows():
    if row['filename'] in filename_to_idx:
        eval_indices.append(filename_to_idx[row['filename']])
        eval_split_labels.append('validation')

for _, row in test_files.iterrows():
    if row['filename'] in filename_to_idx:
        eval_indices.append(filename_to_idx[row['filename']])
        eval_split_labels.append('test')

print(f"✓ Mapped {len(eval_indices)} files to HDF5 indices")

# Create tissue label mapping
unique_tissues = sorted(metadata['tissue_top_level'].unique())
tissue_to_label = {tissue: idx for idx, tissue in enumerate(unique_tissues)}
label_to_tissue = {idx: tissue for tissue, idx in tissue_to_label.items()}

print(f"✓ Found {len(unique_tissues)} unique tissues")
print(f"  Tissues: {', '.join(unique_tissues[:10])}{'...' if len(unique_tissues) > 10 else ''}")

# ============================================================================
# STEP 3: Extract Features and Make Predictions
# ============================================================================

print("\nSTEP 3: Running inference and extracting features...")
print("  This may take a few minutes...")

h5_file = h5py.File(HDF5_PATH, 'r')

all_features = []
all_predictions = []
all_probabilities = []
all_true_labels = []
all_tissue_names = []
all_aug_versions = []
all_sample_names = []
all_split_labels = []

batch_size = 32
n_samples = len(eval_indices)

with torch.no_grad():
    for i in range(0, n_samples, batch_size):
        batch_indices = eval_indices[i:i+batch_size]
        batch_splits = eval_split_labels[i:i+batch_size]
        
        # Load methylation data
        batch_meth = []
        batch_true = []
        batch_tissues = []
        batch_augs = []
        batch_samples = []
        
        for idx in batch_indices:
            meth = h5_file['methylation'][idx]  # [51089, 150]
            
            # Compute region means
            valid_mask = (meth != 2).astype(float)
            region_mean = np.sum(meth * valid_mask, axis=1) / (np.sum(valid_mask, axis=1) + 1e-8)
            
            # Get metadata
            tissue_name = metadata.iloc[idx]['tissue_top_level']
            true_label = tissue_to_label[tissue_name]
            aug_version = metadata.iloc[idx]['aug_version']
            sample_name = metadata.iloc[idx]['sample_name']
            
            batch_meth.append(region_mean)
            batch_true.append(true_label)
            batch_tissues.append(tissue_name)
            batch_augs.append(aug_version)
            batch_samples.append(sample_name)
        
        # Convert to tensors
        batch_meth = torch.tensor(np.array(batch_meth), dtype=torch.float32).to(device)
        
        # Get predictions and features
        logits = model(batch_meth)
        features = model.get_features(batch_meth)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        # Store results
        all_features.append(features.cpu().numpy())
        all_predictions.extend(preds.cpu().numpy())
        all_probabilities.append(probs.cpu().numpy())
        all_true_labels.extend(batch_true)
        all_tissue_names.extend(batch_tissues)
        all_aug_versions.extend(batch_augs)
        all_sample_names.extend(batch_samples)
        all_split_labels.extend(batch_splits)
        
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {i}/{n_samples} samples...")

h5_file.close()

# Concatenate results
all_features = np.concatenate(all_features, axis=0)
all_probabilities = np.concatenate(all_probabilities, axis=0)
all_predictions = np.array(all_predictions)
all_true_labels = np.array(all_true_labels)

print(f"✓ Inference complete!")
print(f"  Features shape: {all_features.shape}")
print(f"  Total samples: {len(all_predictions)}")

# Compute overall accuracy
accuracy = accuracy_score(all_true_labels, all_predictions)
print(f"\n  Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Split accuracies
val_mask = np.array(all_split_labels) == 'validation'
test_mask = np.array(all_split_labels) == 'test'

val_acc = accuracy_score(
    np.array(all_true_labels)[val_mask], 
    np.array(all_predictions)[val_mask]
)
test_acc = accuracy_score(
    np.array(all_true_labels)[test_mask], 
    np.array(all_predictions)[test_mask]
)

print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# ============================================================================
# FIGURE 1: UMAP Embedding (Colored by Tissue Type)
# ============================================================================

print("\nGenerating Figure 1: UMAP embedding...")

# Compute UMAP
print("  Computing UMAP projection (this may take a few minutes)...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
embedding = reducer.fit_transform(all_features)

print("  ✓ UMAP complete")

# Create color map for tissues
tissue_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_tissues)))
if len(unique_tissues) > 20:
    tissue_colors = plt.cm.hsv(np.linspace(0, 1, len(unique_tissues)))

color_map = {tissue: tissue_colors[i] for i, tissue in enumerate(unique_tissues)}

# Plot UMAP
fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# Left: Colored by tissue type
ax = axes[0]
for tissue in unique_tissues:
    mask = np.array(all_tissue_names) == tissue
    if mask.sum() > 0:
        ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                  c=[color_map[tissue]], label=tissue, 
                  alpha=0.6, s=20, edgecolors='none')

ax.set_xlabel('UMAP 1', fontsize=12)
ax.set_ylabel('UMAP 2', fontsize=12)
ax.set_title('UMAP: Features Colored by True Tissue Type', fontsize=14, fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
ax.grid(True, alpha=0.3)

# Right: Colored by prediction correctness
ax = axes[1]
correct_mask = all_predictions == all_true_labels
incorrect_mask = ~correct_mask

ax.scatter(embedding[correct_mask, 0], embedding[correct_mask, 1], 
          c='green', label=f'Correct ({correct_mask.sum()})', 
          alpha=0.6, s=20, edgecolors='none')
ax.scatter(embedding[incorrect_mask, 0], embedding[incorrect_mask, 1], 
          c='red', label=f'Incorrect ({incorrect_mask.sum()})', 
          alpha=0.8, s=30, edgecolors='black', linewidths=0.5)

ax.set_xlabel('UMAP 1', fontsize=12)
ax.set_ylabel('UMAP 2', fontsize=12)
ax.set_title('UMAP: Prediction Correctness', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'fig1_umap_embedding.png'), 
            bbox_inches='tight')
plt.close()
print("  ✓ Saved: fig1_umap_embedding.png")

# ============================================================================
# FIGURE 2: Confusion Matrices (Validation + Test)
# ============================================================================

print("\nGenerating Figure 2: Confusion matrices...")

fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# Overall confusion matrix
ax = axes[0]
cm = confusion_matrix(all_true_labels, all_predictions)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
            xticklabels=unique_tissues, yticklabels=unique_tissues,
            ax=ax, cbar_kws={'label': 'Proportion'})
ax.set_xlabel('Predicted Tissue', fontsize=12)
ax.set_ylabel('True Tissue', fontsize=12)
ax.set_title(f'Overall Confusion Matrix (n={len(all_predictions)})\nAccuracy: {accuracy:.3f}', 
            fontsize=12, fontweight='bold')
plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=8)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

# Validation set
ax = axes[1]
cm_val = confusion_matrix(
    np.array(all_true_labels)[val_mask], 
    np.array(all_predictions)[val_mask]
)
cm_val_normalized = cm_val.astype('float') / (cm_val.sum(axis=1)[:, np.newaxis] + 1e-10)

sns.heatmap(cm_val_normalized, annot=False, fmt='.2f', cmap='Blues',
            xticklabels=unique_tissues, yticklabels=unique_tissues,
            ax=ax, cbar_kws={'label': 'Proportion'})
ax.set_xlabel('Predicted Tissue', fontsize=12)
ax.set_ylabel('True Tissue', fontsize=12)
ax.set_title(f'Validation Set (n={val_mask.sum()})\nAccuracy: {val_acc:.3f}', 
            fontsize=12, fontweight='bold')
plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=8)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

# Test set
ax = axes[2]
cm_test = confusion_matrix(
    np.array(all_true_labels)[test_mask], 
    np.array(all_predictions)[test_mask]
)
cm_test_normalized = cm_test.astype('float') / (cm_test.sum(axis=1)[:, np.newaxis] + 1e-10)

sns.heatmap(cm_test_normalized, annot=False, fmt='.2f', cmap='Blues',
            xticklabels=unique_tissues, yticklabels=unique_tissues,
            ax=ax, cbar_kws={'label': 'Proportion'})
ax.set_xlabel('Predicted Tissue', fontsize=12)
ax.set_ylabel('True Tissue', fontsize=12)
ax.set_title(f'Test Set (n={test_mask.sum()})\nAccuracy: {test_acc:.3f}', 
            fontsize=12, fontweight='bold')
plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=8)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'fig2_confusion_matrices.png'), 
            bbox_inches='tight')
plt.close()
print("  ✓ Saved: fig2_confusion_matrices.png")

# ============================================================================
# FIGURE 3: Per-Tissue Accuracy Heatmap (by Augmentation)
# ============================================================================

print("\nGenerating Figure 3: Per-tissue accuracy by augmentation...")

# Compute accuracy per tissue and augmentation
tissue_aug_accuracy = defaultdict(lambda: defaultdict(list))

for i in range(len(all_predictions)):
    tissue = all_tissue_names[i]
    aug = all_aug_versions[i]
    correct = all_predictions[i] == all_true_labels[i]
    tissue_aug_accuracy[tissue][aug].append(correct)

# Create matrix
aug_versions = sorted(set(all_aug_versions))
accuracy_matrix = np.zeros((len(unique_tissues), len(aug_versions)))

for i, tissue in enumerate(unique_tissues):
    for j, aug in enumerate(aug_versions):
        if aug in tissue_aug_accuracy[tissue] and len(tissue_aug_accuracy[tissue][aug]) > 0:
            accuracy_matrix[i, j] = np.mean(tissue_aug_accuracy[tissue][aug])
        else:
            accuracy_matrix[i, j] = np.nan

# Plot
fig, ax = plt.subplots(figsize=(10, 12))
sns.heatmap(accuracy_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
            xticklabels=[f'Aug {a}' for a in aug_versions],
            yticklabels=unique_tissues,
            ax=ax, cbar_kws={'label': 'Accuracy'},
            vmin=0, vmax=1, linewidths=0.5)
ax.set_xlabel('Augmentation Version', fontsize=12)
ax.set_ylabel('Tissue Type', fontsize=12)
ax.set_title('Per-Tissue Accuracy by Augmentation Version', fontsize=14, fontweight='bold')
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'fig3_accuracy_by_augmentation.png'), 
            bbox_inches='tight')
plt.close()
print("  ✓ Saved: fig3_accuracy_by_augmentation.png")

# ============================================================================
# FIGURE 4: Prediction Confidence Distributions
# ============================================================================

print("\nGenerating Figure 4: Prediction confidence distributions...")

# Get max confidence for each prediction
max_confidences = np.max(all_probabilities, axis=1)
correct_confidences = max_confidences[correct_mask]
incorrect_confidences = max_confidences[incorrect_mask]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Overall confidence distribution
ax = axes[0, 0]
ax.hist(correct_confidences, bins=50, alpha=0.7, label='Correct', 
        color='green', edgecolor='black', density=True)
ax.hist(incorrect_confidences, bins=50, alpha=0.7, label='Incorrect', 
        color='red', edgecolor='black', density=True)
ax.set_xlabel('Prediction Confidence', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xlim([0, 1])

# Confidence vs Accuracy
ax = axes[0, 1]
confidence_bins = np.linspace(0, 1, 11)
bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
bin_accuracies = []

for i in range(len(confidence_bins) - 1):
    mask = (max_confidences >= confidence_bins[i]) & (max_confidences < confidence_bins[i+1])
    if mask.sum() > 0:
        bin_acc = (all_predictions[mask] == all_true_labels[mask]).mean()
        bin_accuracies.append(bin_acc)
    else:
        bin_accuracies.append(np.nan)

ax.plot(bin_centers, bin_accuracies, 'o-', linewidth=2, markersize=8, label='Actual')
ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect calibration')
ax.set_xlabel('Prediction Confidence', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Calibration Curve\n(Confidence vs Actual Accuracy)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

# Per-tissue confidence
ax = axes[1, 0]
tissue_confidences = defaultdict(list)
for i in range(len(all_predictions)):
    if all_predictions[i] == all_true_labels[i]:
        tissue_confidences[all_tissue_names[i]].append(max_confidences[i])

tissue_names_sorted = sorted(tissue_confidences.keys(), 
                             key=lambda t: np.mean(tissue_confidences[t]) if len(tissue_confidences[t]) > 0 else 0,
                             reverse=True)
tissue_means = [np.mean(tissue_confidences[t]) if len(tissue_confidences[t]) > 0 else 0 
               for t in tissue_names_sorted]

bars = ax.barh(range(len(tissue_names_sorted)), tissue_means, alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(tissue_names_sorted)))
ax.set_yticklabels(tissue_names_sorted, fontsize=8)
ax.set_xlabel('Mean Confidence (Correct Predictions)', fontsize=12)
ax.set_title('Per-Tissue Prediction Confidence', fontsize=12, fontweight='bold')
ax.set_xlim([0, 1])
ax.grid(True, alpha=0.3, axis='x')

# Color bars by value
for i, bar in enumerate(bars):
    bar.set_color(plt.cm.RdYlGn(tissue_means[i]))

# Confidence by augmentation version
ax = axes[1, 1]
aug_confidences = defaultdict(list)
for i in range(len(all_predictions)):
    if all_predictions[i] == all_true_labels[i]:
        aug_confidences[all_aug_versions[i]].append(max_confidences[i])

aug_sorted = sorted(aug_confidences.keys())
aug_data = [aug_confidences[aug] for aug in aug_sorted]

bp = ax.boxplot(aug_data, labels=[f'Aug {a}' for a in aug_sorted],
               patch_artist=True, showmeans=True)

for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)

ax.set_xlabel('Augmentation Version', fontsize=12)
ax.set_ylabel('Prediction Confidence', fontsize=12)
ax.set_title('Confidence by Augmentation Version', fontsize=12, fontweight='bold')
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'fig4_prediction_confidence.png'), 
            bbox_inches='tight')
plt.close()
print("  ✓ Saved: fig4_prediction_confidence.png")

# ============================================================================
# FIGURE 5: Misclassification Analysis
# ============================================================================

print("\nGenerating Figure 5: Misclassification analysis...")

# Get misclassification patterns
misclass_matrix = np.zeros((len(unique_tissues), len(unique_tissues)))

for i in range(len(all_predictions)):
    if all_predictions[i] != all_true_labels[i]:
        true_idx = all_true_labels[i]
        pred_idx = all_predictions[i]
        misclass_matrix[true_idx, pred_idx] += 1

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Heatmap of misclassifications
ax = axes[0]
sns.heatmap(misclass_matrix, annot=False, cmap='Reds',
            xticklabels=unique_tissues, yticklabels=unique_tissues,
            ax=ax, cbar_kws={'label': 'Count'})
ax.set_xlabel('Predicted Tissue', fontsize=12)
ax.set_ylabel('True Tissue', fontsize=12)
ax.set_title('Misclassification Patterns\n(Off-diagonal = errors)', 
            fontsize=12, fontweight='bold')
plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=8)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

# Top misclassification pairs
ax = axes[1]
misclass_pairs = []
for i in range(len(unique_tissues)):
    for j in range(len(unique_tissues)):
        if i != j and misclass_matrix[i, j] > 0:
            misclass_pairs.append({
                'true': unique_tissues[i],
                'pred': unique_tissues[j],
                'count': misclass_matrix[i, j]
            })

misclass_pairs = sorted(misclass_pairs, key=lambda x: x['count'], reverse=True)[:15]

if len(misclass_pairs) > 0:
    labels = [f"{p['true']} → {p['pred']}" for p in misclass_pairs]
    counts = [p['count'] for p in misclass_pairs]
    
    bars = ax.barh(range(len(labels)), counts, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Number of Misclassifications', fontsize=12)
    ax.set_title('Top 15 Misclassification Pairs', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Color bars
    for bar in bars:
        bar.set_color('salmon')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'fig5_misclassification_analysis.png'), 
            bbox_inches='tight')
plt.close()
print("  ✓ Saved: fig5_misclassification_analysis.png")

# ============================================================================
# FIGURE 6: Per-Tissue Performance Metrics
# ============================================================================

print("\nGenerating Figure 6: Per-tissue performance metrics...")

# Compute per-tissue metrics
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(
    all_true_labels, all_predictions, labels=range(len(unique_tissues)), zero_division=0
)

# Create dataframe
metrics_df = pd.DataFrame({
    'Tissue': unique_tissues,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
})

# Sort by F1-score
metrics_df = metrics_df.sort_values('F1-Score', ascending=True)

fig, axes = plt.subplots(1, 3, figsize=(20, 10))

# Precision
ax = axes[0]
bars = ax.barh(range(len(metrics_df)), metrics_df['Precision'].values, 
              alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(metrics_df)))
ax.set_yticklabels(metrics_df['Tissue'].values, fontsize=9)
ax.set_xlabel('Precision', fontsize=12)
ax.set_title('Per-Tissue Precision', fontsize=12, fontweight='bold')
ax.set_xlim([0, 1])
ax.grid(True, alpha=0.3, axis='x')
for i, bar in enumerate(bars):
    bar.set_color(plt.cm.RdYlGn(metrics_df['Precision'].values[i]))

# Recall
ax = axes[1]
bars = ax.barh(range(len(metrics_df)), metrics_df['Recall'].values, 
              alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(metrics_df)))
ax.set_yticklabels(metrics_df['Tissue'].values, fontsize=9)
ax.set_xlabel('Recall', fontsize=12)
ax.set_title('Per-Tissue Recall', fontsize=12, fontweight='bold')
ax.set_xlim([0, 1])
ax.grid(True, alpha=0.3, axis='x')
for i, bar in enumerate(bars):
    bar.set_color(plt.cm.RdYlGn(metrics_df['Recall'].values[i]))

# F1-Score
ax = axes[2]
bars = ax.barh(range(len(metrics_df)), metrics_df['F1-Score'].values, 
              alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(metrics_df)))
ax.set_yticklabels(metrics_df['Tissue'].values, fontsize=9)
ax.set_xlabel('F1-Score', fontsize=12)
ax.set_title('Per-Tissue F1-Score', fontsize=12, fontweight='bold')
ax.set_xlim([0, 1])
ax.grid(True, alpha=0.3, axis='x')
for i, bar in enumerate(bars):
    bar.set_color(plt.cm.RdYlGn(metrics_df['F1-Score'].values[i]))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'fig6_per_tissue_metrics.png'), 
            bbox_inches='tight')
plt.close()
print("  ✓ Saved: fig6_per_tissue_metrics.png")

# ============================================================================
# FIGURE 7: Augmentation Consistency Analysis
# ============================================================================

print("\nGenerating Figure 7: Augmentation consistency analysis...")

# For each unique sample, check if predictions are consistent across augmentations
sample_consistency = defaultdict(lambda: {'tissues': set(), 'predictions': [], 'correct': []})

for i in range(len(all_predictions)):
    sample_name = all_sample_names[i]
    tissue = all_tissue_names[i]
    pred = all_predictions[i]
    correct = all_predictions[i] == all_true_labels[i]
    
    sample_consistency[sample_name]['tissues'].add(tissue)
    sample_consistency[sample_name]['predictions'].append(pred)
    sample_consistency[sample_name]['correct'].append(correct)

# Compute consistency score
consistency_scores = []
tissue_labels = []

for sample_name, data in sample_consistency.items():
    if len(data['predictions']) > 1:  # Multiple augmentations
        # All predictions same?
        unique_preds = set(data['predictions'])
        consistency = 1.0 if len(unique_preds) == 1 else len(data['correct']) / len(data['predictions'])
        consistency_scores.append(consistency)
        tissue_labels.append(list(data['tissues'])[0])

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Overall consistency distribution
ax = axes[0]
ax.hist(consistency_scores, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
ax.set_xlabel('Consistency Score Across Augmentations', fontsize=12)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_title('Prediction Consistency Across Augmentation Versions\n(1.0 = all augmentations predicted same)', 
            fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.axvline(x=np.mean(consistency_scores), color='red', linestyle='--', 
          linewidth=2, label=f'Mean: {np.mean(consistency_scores):.3f}')
ax.legend(fontsize=10)

# Per-tissue consistency
ax = axes[1]
tissue_consistency = defaultdict(list)
for cons, tissue in zip(consistency_scores, tissue_labels):
    tissue_consistency[tissue].append(cons)

tissue_names_cons = sorted(tissue_consistency.keys(), 
                          key=lambda t: np.mean(tissue_consistency[t]),
                          reverse=False)
tissue_means_cons = [np.mean(tissue_consistency[t]) for t in tissue_names_cons]

bars = ax.barh(range(len(tissue_names_cons)), tissue_means_cons, 
              alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(tissue_names_cons)))
ax.set_yticklabels(tissue_names_cons, fontsize=9)
ax.set_xlabel('Mean Consistency Score', fontsize=12)
ax.set_title('Per-Tissue Augmentation Consistency', fontsize=12, fontweight='bold')
ax.set_xlim([0, 1])
ax.grid(True, alpha=0.3, axis='x')
ax.axvline(x=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5)

for i, bar in enumerate(bars):
    bar.set_color(plt.cm.RdYlGn(tissue_means_cons[i]))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'fig7_augmentation_consistency.png'), 
            bbox_inches='tight')
plt.close()
print("  ✓ Saved: fig7_augmentation_consistency.png")

# ============================================================================
# Save Summary Report
# ============================================================================

print("\nGenerating summary report...")

report_lines = []
report_lines.append("="*80)
report_lines.append("SINGLE-TISSUE MODEL PERFORMANCE REPORT")
report_lines.append("Validation + Test Sets (All Augmentations)")
report_lines.append("="*80)
report_lines.append("")

report_lines.append("1. DATASET SUMMARY")
report_lines.append("-" * 80)
report_lines.append(f"Total samples evaluated: {len(all_predictions)}")
report_lines.append(f"  Validation set: {val_mask.sum()}")
report_lines.append(f"  Test set: {test_mask.sum()}")
report_lines.append(f"Number of tissues: {len(unique_tissues)}")
report_lines.append(f"Augmentation versions: {sorted(set(all_aug_versions))}")
report_lines.append("")

report_lines.append("2. OVERALL PERFORMANCE")
report_lines.append("-" * 80)
report_lines.append(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
report_lines.append(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
report_lines.append(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
report_lines.append(f"Correct predictions: {correct_mask.sum()}")
report_lines.append(f"Incorrect predictions: {incorrect_mask.sum()}")
report_lines.append("")

report_lines.append("3. CONFIDENCE ANALYSIS")
report_lines.append("-" * 80)
report_lines.append(f"Mean confidence (correct): {np.mean(correct_confidences):.4f}")
report_lines.append(f"Mean confidence (incorrect): {np.mean(incorrect_confidences):.4f}")
report_lines.append(f"Confidence gap: {np.mean(correct_confidences) - np.mean(incorrect_confidences):.4f}")
report_lines.append("")

report_lines.append("4. TOP 5 BEST PERFORMING TISSUES")
report_lines.append("-" * 80)
best_tissues = metrics_df.nlargest(5, 'F1-Score')
for _, row in best_tissues.iterrows():
    report_lines.append(f"  {row['Tissue']:20s}: F1={row['F1-Score']:.3f}, "
                       f"Precision={row['Precision']:.3f}, Recall={row['Recall']:.3f}, "
                       f"n={int(row['Support'])}")
report_lines.append("")

report_lines.append("5. TOP 5 WORST PERFORMING TISSUES")
report_lines.append("-" * 80)
worst_tissues = metrics_df.nsmallest(5, 'F1-Score')
for _, row in worst_tissues.iterrows():
    report_lines.append(f"  {row['Tissue']:20s}: F1={row['F1-Score']:.3f}, "
                       f"Precision={row['Precision']:.3f}, Recall={row['Recall']:.3f}, "
                       f"n={int(row['Support'])}")
report_lines.append("")

report_lines.append("6. AUGMENTATION CONSISTENCY")
report_lines.append("-" * 80)
report_lines.append(f"Mean consistency score: {np.mean(consistency_scores):.4f}")
report_lines.append(f"Samples with perfect consistency: {(np.array(consistency_scores) == 1.0).sum()}")
report_lines.append("")

report_lines.append("7. TOP MISCLASSIFICATION PAIRS")
report_lines.append("-" * 80)
for i, pair in enumerate(misclass_pairs[:10]):
    report_lines.append(f"  {i+1}. {pair['true']:15s} → {pair['pred']:15s}: {int(pair['count'])} errors")
report_lines.append("")

report_lines.append("="*80)
report_lines.append("FIGURES GENERATED:")
report_lines.append("  1. fig1_umap_embedding.png - UMAP visualization")
report_lines.append("  2. fig2_confusion_matrices.png - Confusion matrices")
report_lines.append("  3. fig3_accuracy_by_augmentation.png - Per-tissue accuracy by aug")
report_lines.append("  4. fig4_prediction_confidence.png - Confidence distributions")
report_lines.append("  5. fig5_misclassification_analysis.png - Error patterns")
report_lines.append("  6. fig6_per_tissue_metrics.png - Precision/Recall/F1")
report_lines.append("  7. fig7_augmentation_consistency.png - Consistency analysis")
report_lines.append("="*80)

# Print and save report
print("\n" + "\n".join(report_lines))

report_path = os.path.join(OUTPUT_DIR, 'performance_report.txt')
with open(report_path, 'w') as f:
    f.write("\n".join(report_lines))

# Save metrics CSV
metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'per_tissue_metrics.csv'), index=False)

print(f"\n✓ Saved report: {report_path}")
print(f"✓ Saved metrics: per_tissue_metrics.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
print(f"\nAll results saved to: {OUTPUT_DIR}")
print("\nGenerated 7 comprehensive figures:")
print("  1. UMAP embedding with tissue colors")
print("  2. Confusion matrices (overall, val, test)")
print("  3. Per-tissue accuracy by augmentation heatmap")
print("  4. Prediction confidence analysis")
print("  5. Misclassification patterns")
print("  6. Per-tissue precision/recall/F1 metrics")
print("  7. Augmentation consistency analysis")
print("\nSummary files:")
print("  - performance_report.txt")
print("  - per_tissue_metrics.csv")
print("\n" + "="*80)
