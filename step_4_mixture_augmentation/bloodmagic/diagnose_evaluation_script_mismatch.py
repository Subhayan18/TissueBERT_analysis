#!/usr/bin/env python3
"""
Diagnostic: Check Stage 2 Test Data and Model Outputs
======================================================
"""

import torch
import h5py
import numpy as np
import sys

print("="*80)
print("STAGE 2 EVALUATION DIAGNOSTIC")
print("="*80)

# 1. Check test data file
test_path = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/mixture_data/stage2_validation_absolute.h5'

print("\n1. CHECKING TEST DATA FILE")
print("-"*80)
try:
    with h5py.File(test_path, 'r') as f:
        props = f['true_proportions'][:]
        meth = f['mixed_methylation'][:]
        sums = props.sum(axis=1)
        
        print(f"File: {test_path}")
        print(f"  ✓ EXISTS")
        print(f"  Shape: {props.shape}")
        print(f"  Mixed methylation: {meth.shape}")
        print()
        print(f"  Label statistics:")
        print(f"    Mean sum: {sums.mean():.4f}")
        print(f"    Std sum: {sums.std():.4f}")
        print(f"    Min sum: {sums.min():.4f}")
        print(f"    Max sum: {sums.max():.4f}")
        print()
        
        if np.allclose(sums, 1.0, atol=0.01):
            print("  ❌ PROBLEM: Labels are RENORMALIZED (sum ≈ 1.0)")
            print("     These are OLD labels from broken training!")
            print("     Model trained with absolute labels, test data has renormalized labels")
            print("     → This mismatch causes poor metrics")
        elif sums.mean() > 0.05 and sums.mean() < 0.45:
            print("  ✓ CORRECT: Labels are ABSOLUTE")
            print(f"     Labels sum to {sums.mean():.3f} (representing non-blood proportion)")
        else:
            print(f"  ⚠️  UNEXPECTED: Label sums are {sums.mean():.3f}")
            
except FileNotFoundError:
    print(f"  ❌ NOT FOUND: {test_path}")
    print("     You need to generate test data first!")
    sys.exit(1)

# 2. Check model checkpoint
checkpoint_path = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/stage2_bloodmasked_extended/checkpoints/checkpoint_best.pt'

print("\n2. CHECKING MODEL CHECKPOINT")
print("-"*80)
try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"File: {checkpoint_path}")
    print(f"  ✓ EXISTS")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Val MAE: {checkpoint.get('val_mae', 'unknown')}")
    
    # Check if model has normalize_output parameter
    state_dict = checkpoint['model_state_dict']
    print(f"  Model parameters: {len(state_dict)} tensors")
    
except FileNotFoundError:
    print(f"  ❌ NOT FOUND: {checkpoint_path}")
    sys.exit(1)

# 3. Test model loading and output
print("\n3. TESTING MODEL OUTPUT")
print("-"*80)

sys.path.append('/home/chattopa/data_storage/TissueBERT_analysis/step_4_mixture_augmentation/bloodmagic')

try:
    from model_deconvolution_absolute import TissueBERTDeconvolution
    print("  ✓ Using model_deconvolution_absolute.py")
    has_absolute_model = True
except ImportError:
    from model_deconvolution import TissueBERTDeconvolution
    print("  ⚠️  Using model_deconvolution.py (may not have normalize_output)")
    has_absolute_model = False

# Test with normalize_output=False
print("\n  Testing with normalize_output=False:")
try:
    model_abs = TissueBERTDeconvolution(
        num_classes=21,
        n_regions=51089,
        normalize_output=False
    )
    model_abs.load_state_dict(checkpoint['model_state_dict'])
    model_abs.eval()
    
    dummy_input = torch.randn(2, 51089, 150)
    with torch.no_grad():
        output_abs = model_abs(dummy_input)
    
    sums_abs = output_abs.sum(dim=1)
    print(f"    Output shape: {output_abs.shape}")
    print(f"    Output sums: {sums_abs.numpy()}")
    print(f"    Mean sum: {sums_abs.mean().item():.4f}")
    
    if abs(sums_abs.mean().item() - 1.0) < 0.01:
        print("    ❌ Outputs sum to 1.0 (normalized) - wrong!")
    else:
        print("    ✓ Outputs are absolute (not normalized)")
        
except TypeError as e:
    print(f"    ❌ ERROR: model_deconvolution.py doesn't support normalize_output")
    print(f"       You need to use model_deconvolution_absolute.py")

# Test with default (should normalize)
print("\n  Testing with default settings:")
model_default = TissueBERTDeconvolution(
    num_classes=21,
    n_regions=51089
)
model_default.load_state_dict(checkpoint['model_state_dict'])
model_default.eval()

with torch.no_grad():
    output_default = model_default(dummy_input)

sums_default = output_default.sum(dim=1)
print(f"    Output shape: {output_default.shape}")
print(f"    Output sums: {sums_default.numpy()}")
print(f"    Mean sum: {sums_default.mean().item():.4f}")

if abs(sums_default.mean().item() - 1.0) < 0.01:
    print("    ⚠️  Outputs sum to 1.0 (normalized by default)")
else:
    print("    Output sums:", sums_default.mean().item())

# 4. Diagnosis
print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

# Check if there's a mismatch
with h5py.File(test_path, 'r') as f:
    test_label_sum = f['true_proportions'][:].sum(axis=1).mean()

model_output_sum = sums_default.mean().item()

print(f"\nTest data label sum: {test_label_sum:.4f}")
print(f"Model output sum: {model_output_sum:.4f}")
print()

if np.allclose(test_label_sum, 1.0, atol=0.01) and abs(model_output_sum - 1.0) < 0.01:
    print("❌ PROBLEM IDENTIFIED:")
    print("   Both test data AND model are normalized (sum=1.0)")
    print("   BUT the model was TRAINED with absolute labels!")
    print()
    print("   This means:")
    print("   1. Test data has OLD renormalized labels")
    print("   2. Model outputs are being normalized (wrong!)")
    print()
    print("   SOLUTIONS:")
    print("   A. Generate NEW test data with absolute labels:")
    print("      sbatch generate_stage2_datasets.sh")
    print()
    print("   B. Use model_deconvolution_absolute with normalize_output=False")
    print("      in evaluate_deconvolution_stage2.py line 54")
    
elif not np.allclose(test_label_sum, 1.0, atol=0.01) and abs(model_output_sum - 1.0) < 0.01:
    print("❌ MISMATCH:")
    print(f"   Test labels are absolute (sum={test_label_sum:.3f})")
    print(f"   But model outputs are normalized (sum=1.0)")
    print()
    print("   FIX: Use normalize_output=False in evaluation script")
    
elif np.allclose(test_label_sum, 1.0, atol=0.01) and abs(model_output_sum - 1.0) > 0.1:
    print("❌ MISMATCH:")
    print(f"   Test labels are normalized (sum=1.0)")
    print(f"   But model outputs are absolute (sum={model_output_sum:.3f})")
    print()
    print("   FIX: Generate new test data with absolute labels")
    
else:
    print("✓ BOTH are absolute - this should work!")
    print(f"   Test label sum: {test_label_sum:.3f}")
    print(f"   Model output sum: {model_output_sum:.3f}")

print("="*80)
