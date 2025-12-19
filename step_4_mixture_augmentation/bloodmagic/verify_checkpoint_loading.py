#!/usr/bin/env python3
"""
Test Stage 2 Checkpoint Loading
================================

Quick test to verify Phase 3 checkpoint can be loaded into Stage 2 model.

Usage:
    python test_checkpoint_loading.py
"""

import sys
import torch
from pathlib import Path

print("="*80)
print("STAGE 2 CHECKPOINT LOADING TEST")
print("="*80)

# Import model
try:
    from model_deconvolution import TissueBERTDeconvolution
    print("✓ Imported TissueBERTDeconvolution")
except ImportError as e:
    print(f"✗ Cannot import model_deconvolution: {e}")
    sys.exit(1)

# Paths
PHASE3_CKPT = '/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase3_realistic/checkpoints/checkpoint_best.pt'

# Check Phase 3 checkpoint exists
print(f"\nPhase 3 checkpoint:")
print(f"  Path: {PHASE3_CKPT}")
if not Path(PHASE3_CKPT).exists():
    print(f"  ✗ NOT FOUND")
    sys.exit(1)
else:
    size_mb = Path(PHASE3_CKPT).stat().st_size / (1024 * 1024)
    print(f"  ✓ Found ({size_mb:.1f} MB)")

# Load Phase 3 checkpoint
print(f"\nLoading Phase 3 checkpoint...")
checkpoint = torch.load(PHASE3_CKPT, map_location='cpu')
print(f"  Checkpoint keys: {list(checkpoint.keys())}")

phase3_state = checkpoint['model_state_dict']
print(f"  Model state dict keys: {len(phase3_state)}")

# Check Phase 3 config
if 'config' in checkpoint:
    phase3_classes = checkpoint['config']['model']['num_classes']
    print(f"  Phase 3 num_classes: {phase3_classes}")
else:
    print(f"  Warning: No config in checkpoint, assuming 22 classes")
    phase3_classes = 22

# Create Stage 2 model (21 classes)
print(f"\n" + "="*80)
print(f"Creating Stage 2 model (21 classes)...")
print(f"="*80)

try:
    model_stage2 = TissueBERTDeconvolution(
        n_regions=51089,
        hidden_size=512,
        num_classes=21,  # Stage 2: Blood removed
        dropout=0.1,
        intermediate_size=2048
    )
    print(f"✓ Created Stage 2 model")
    
    # Count parameters
    n_params = sum(p.numel() for p in model_stage2.parameters())
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
except Exception as e:
    print(f"✗ Failed to create model: {e}")
    sys.exit(1)

# Get Stage 2 model state
stage2_state = model_stage2.state_dict()
print(f"  State dict keys: {len(stage2_state)}")

# Compare architectures
print(f"\n" + "="*80)
print(f"Comparing Phase 3 vs Stage 2 architectures...")
print(f"="*80)

compatible_count = 0
incompatible_count = 0
incompatible_keys = []

for key in phase3_state.keys():
    if key not in stage2_state:
        print(f"  Key in Phase 3 but not in Stage 2: {key}")
        incompatible_count += 1
        continue
    
    phase3_shape = phase3_state[key].shape
    stage2_shape = stage2_state[key].shape
    
    if phase3_shape != stage2_shape:
        incompatible_keys.append((key, phase3_shape, stage2_shape))
        incompatible_count += 1
    else:
        compatible_count += 1

print(f"\n  Compatible parameters: {compatible_count}")
print(f"  Incompatible parameters: {incompatible_count}")

if incompatible_keys:
    print(f"\n  Incompatible keys (shape mismatch):")
    for key, p3_shape, s2_shape in incompatible_keys[:10]:
        print(f"    {key:40s} Phase3:{str(p3_shape):15s} Stage2:{str(s2_shape):15s}")
    if len(incompatible_keys) > 10:
        print(f"    ... and {len(incompatible_keys) - 10} more")

# Attempt to load compatible weights
print(f"\n" + "="*80)
print(f"Loading compatible weights...")
print(f"="*80)

compatible_state = {}
for key, value in phase3_state.items():
    if key in stage2_state and value.shape == stage2_state[key].shape:
        compatible_state[key] = value

try:
    missing_keys, unexpected_keys = model_stage2.load_state_dict(compatible_state, strict=False)
    
    print(f"✓ Successfully loaded compatible weights")
    print(f"  Loaded: {len(compatible_state)} parameters")
    print(f"  Missing (newly initialized): {len(missing_keys)}")
    print(f"  Unexpected: {len(unexpected_keys)}")
    
    if missing_keys:
        print(f"\n  Missing keys (will use random initialization):")
        for key in missing_keys[:5]:
            print(f"    - {key}")
        if len(missing_keys) > 5:
            print(f"    ... and {len(missing_keys) - 5} more")
    
    # Test forward pass
    print(f"\n" + "="*80)
    print(f"Testing forward pass...")
    print(f"="*80)
    
    dummy_input = torch.randn(2, 51089, 150)  # Batch of 2
    print(f"  Input shape: {dummy_input.shape}")
    
    model_stage2.eval()
    with torch.no_grad():
        output = model_stage2(dummy_input)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Output sum (should be ~1.0): {output.sum(dim=1)}")
    print(f"✓ Forward pass successful!")
    
except Exception as e:
    print(f"✗ Failed to load weights or forward pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final summary
print(f"\n" + "="*80)
print(f"TEST SUMMARY")
print(f"="*80)
print(f"✓ Phase 3 checkpoint loaded successfully")
print(f"✓ Stage 2 model created with 21 classes")
print(f"✓ {len(compatible_state)} compatible parameters transferred")
print(f"✓ Forward pass works correctly")
print(f"\n➜ Stage 2 training should work correctly")
print(f"  Run: sbatch submit_stage2.sh")

print("="*80)
