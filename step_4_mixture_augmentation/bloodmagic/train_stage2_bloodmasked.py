#!/usr/bin/env python3
"""
Stage 2 Blood-Masked Tissue Deconvolution Training Script
==========================================================

Trains model to predict NON-BLOOD tissue proportions from blood-dominant mixtures.

Key differences from Phase 3:
- Model output: 21 tissues (Blood removed)
- Labels: Blood-masked, renormalized
- Goal: Learn to ignore blood signal and detect trace tissue signals

This fine-tunes from Phase 3 checkpoint with modified classification head.

Author: Stage 2 Blood Deconvolution
Date: December 2024
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print(f"Python path: {sys.path[:3]}")
print(f"Current directory: {current_dir}")
print(f"Files in directory: {os.listdir(current_dir)}")

# Import from existing training script
try:
    from train_deconvolution import DeconvolutionTrainer
    print("✓ Successfully imported train_deconvolution")
except ImportError as e:
    print(f"ERROR: Cannot import train_deconvolution.py")
    print(f"  ImportError: {e}")
    print(f"  Current directory: {os.getcwd()}")
    print(f"  Python path: {sys.path}")
    print(f"  Files in current directory: {os.listdir('.')}")
    print("\n  Please ensure train_deconvolution.py is in the same directory")
    sys.exit(1)

# Import blood-masked dataloader
try:
    from dataloader_mixture_stage2 import create_bloodmasked_dataloaders
    print("✓ Successfully imported dataloader_mixture_stage2")
except ImportError as e:
    print(f"ERROR: Cannot import dataloader_mixture_stage2.py")
    print(f"  ImportError: {e}")
    print(f"  Please ensure dataloader_mixture_stage2.py is in the same directory")
    sys.exit(1)


class Stage2Trainer(DeconvolutionTrainer):
    """
    Extended trainer for Stage 2 blood-masked training
    
    Overrides model building and dataloader creation
    """
    
    def build_model(self):
        """
        Build Stage 2 model with 21 classes (Blood removed)
        
        CRITICAL: Do NOT use load_pretrained_model() because it hardcodes num_classes=22
        Instead, build fresh model and manually load weights later
        """
        print("\n" + "="*80)
        print("Building Stage 2 Model")
        print("="*80)
        
        model_config = self.config['model']
        
        # FORCE num_classes to 21
        if model_config['num_classes'] != 21:
            print(f"WARNING: Config has num_classes={model_config['num_classes']}, forcing to 21")
            model_config['num_classes'] = 21
        
        print(f"\nModel configuration:")
        print(f"  n_regions: {model_config['n_regions']}")
        print(f"  hidden_size: {model_config['hidden_size']}")
        print(f"  num_classes: {model_config['num_classes']} (Blood removed)")
        print(f"  dropout: {model_config['dropout']}")
        print(f"  intermediate_size: {model_config['intermediate_size']}")
        print(f"  normalize_output: False (absolute labels)")
        
        # Import model class - try multiple methods
        import sys
        import os
        from pathlib import Path
        
        # Get the directory containing this script
        script_dir = Path(__file__).parent.absolute()
        
        # Add to path if not already there
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        
        # Try importing
        try:
            from model_deconvolution_absolute import TissueBERTDeconvolution
            print(f"  ✓ Imported from: {script_dir}/model_deconvolution_absolute.py")
        except ImportError:
            # Fallback: try loading from explicit path
            import importlib.util
            model_path = script_dir / 'model_deconvolution_absolute.py'
            if not model_path.exists():
                raise FileNotFoundError(
                    f"ERROR: model_deconvolution_absolute.py not found!\n"
                    f"  Expected at: {model_path}\n"
                    f"  Please copy from /mnt/user-data/outputs/model_deconvolution_absolute.py"
                )
            
            spec = importlib.util.spec_from_file_location("model_deconvolution_absolute", model_path)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            TissueBERTDeconvolution = model_module.TissueBERTDeconvolution
            print(f"  ✓ Loaded from: {model_path}")
        
        # Build fresh model with 21 classes
        # DO NOT call load_pretrained_model() - it hardcodes num_classes=22
        self.model = TissueBERTDeconvolution(
            vocab_size=model_config.get('vocab_size', 69),
            hidden_size=model_config['hidden_size'],
            num_hidden_layers=model_config.get('num_hidden_layers', 3),
            num_attention_heads=model_config.get('num_attention_heads', 4),
            intermediate_size=model_config['intermediate_size'],
            max_position_embeddings=model_config.get('max_position_embeddings', 150),
            num_classes=21,  # HARDCODE 21 for Stage 2
            dropout=model_config['dropout'],
            n_regions=model_config['n_regions'],
            normalize_output=False  # CRITICAL: Use absolute labels, not normalized
        ).to(self.device)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n✓ Model created successfully")
        print(f"  Trainable parameters: {n_params:,} ({n_params/1e6:.2f}M)")
        
        # Verify output dimension with dummy forward pass
        import torch
        dummy_input = torch.randn(1, 51089, 150).to(self.device)
        self.model.eval()
        with torch.no_grad():
            dummy_output = self.model(dummy_input)
        self.model.train()
        
        print(f"\nVerification:")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {dummy_output.shape}")
        print(f"  Output sum: {dummy_output.sum(dim=1).item():.4f}")
        
        if dummy_output.shape[1] != 21:
            raise ValueError(
                f"ERROR: Model outputs {dummy_output.shape[1]} classes, expected 21!\n"
                f"This should never happen. Check TissueBERTDeconvolution initialization."
            )
        
        output_sum = dummy_output.sum(dim=1).item()
        if abs(output_sum - 1.0) < 0.01:
            raise ValueError(
                f"ERROR: Model outputs sum to 1.0 (normalized), but we need absolute labels!\n"
                f"  Output sum: {output_sum:.4f}\n"
                f"  This means normalize_output=True, but it should be False for Stage 2.\n"
                f"  Check that model_deconvolution_absolute.py is being used."
            )
        
        print(f"  ✓ Model correctly outputs 21 classes")
        print(f"  ✓ Outputs are absolute (sum={output_sum:.4f} != 1.0)")
        
        # NOTE: Phase 3 checkpoint will be loaded AFTER this in load_pretrained_checkpoint()
    
    def build_dataloaders(self):
        """Build Stage 2 blood-masked dataloaders"""
        print("\n" + "="*80)
        print("Building Stage 2 Blood-Masked DataLoaders")
        print("="*80)
        
        data_config = self.config['data']
        training_config = self.config['training']
        
        self.train_loader, self.val_loader, self.test_loader = create_bloodmasked_dataloaders(
            hdf5_path=data_config['hdf5_path'],
            metadata_csv=data_config['metadata_csv'],
            validation_h5=data_config['validation_h5'],
            test_h5=data_config['test_h5'],
            batch_size=training_config['batch_size'],
            n_mixtures_per_epoch=training_config['mixtures_per_epoch'],
            pure_sample_ratio=training_config['pure_sample_ratio'],
            num_workers=training_config['num_workers'],
            seed=self.config['random_seed']
        )
    
    def load_pretrained_checkpoint(self):
        """
        Load Phase 3 checkpoint and adapt for Stage 2
        
        Phase 3: num_classes=22 (including Blood)
        Stage 2: num_classes=21 (Blood removed)
        
        Strategy: Load encoder weights, reinitialize classification head
        """
        pretrained_path = self.config['model'].get('pretrained_checkpoint')
        
        if pretrained_path is None or not Path(pretrained_path).exists():
            print(f"\n⚠️  No pretrained checkpoint found at: {pretrained_path}")
            print("  Training from scratch (not recommended for Stage 2)")
            return
        
        print(f"\n" + "="*80)
        print(f"Loading Phase 3 checkpoint for Stage 2 fine-tuning")
        print(f"="*80)
        print(f"Checkpoint: {pretrained_path}")
        
        import torch
        checkpoint = torch.load(pretrained_path, map_location=self.device)
        
        # Get model state
        model_state = checkpoint['model_state_dict']
        
        print(f"\nPhase 3 checkpoint info:")
        if 'config' in checkpoint:
            phase3_classes = checkpoint['config']['model'].get('num_classes', 'unknown')
            print(f"  Phase 3 num_classes: {phase3_classes}")
        print(f"  Stage 2 num_classes: {self.config['model']['num_classes']}")
        
        # Load compatible weights (skip incompatible classifier)
        compatible_state = {}
        incompatible_keys = []
        skipped_keys = []
        
        current_model_dict = self.model.state_dict()
        
        for key, value in model_state.items():
            # Check if key exists in current model
            if key not in current_model_dict:
                skipped_keys.append(key)
                continue
            
            # Check if shapes match
            if value.shape != current_model_dict[key].shape:
                incompatible_keys.append(f"{key}: {value.shape} vs {current_model_dict[key].shape}")
                print(f"  Skipping {key}: shape mismatch {value.shape} vs {current_model_dict[key].shape}")
                continue
            
            # Load compatible weight
            compatible_state[key] = value
        
        # Load compatible weights with strict=False
        missing_keys, unexpected_keys = self.model.load_state_dict(compatible_state, strict=False)
        
        print(f"\n✓ Loaded encoder weights from Phase 3")
        print(f"  Loaded parameters: {len(compatible_state)}")
        print(f"  Missing keys (newly initialized): {len(missing_keys)}")
        print(f"  Incompatible keys (shape mismatch): {len(incompatible_keys)}")
        
        if missing_keys:
            print(f"\n  Missing keys (will use random initialization):")
            for key in missing_keys[:5]:  # Show first 5
                print(f"    - {key}")
            if len(missing_keys) > 5:
                print(f"    ... and {len(missing_keys) - 5} more")
        
        if incompatible_keys:
            print(f"\n  Incompatible keys (skipped):")
            for key in incompatible_keys[:5]:
                print(f"    - {key}")
            if len(incompatible_keys) > 5:
                print(f"    ... and {len(incompatible_keys) - 5} more")
        
        # Don't load optimizer/scheduler (fresh start for fine-tuning)
        print(f"\n  Starting fresh optimizer for Stage 2 fine-tuning")
        print(f"  Epoch counter reset to 0")
        
        # CRITICAL: Verify model still outputs 21 classes after loading
        print(f"\n" + "="*80)
        print(f"Verifying model architecture after checkpoint loading")
        print(f"="*80)
        
        import torch
        dummy_input = torch.randn(1, 51089, 150).to(self.device)
        self.model.eval()
        with torch.no_grad():
            dummy_output = self.model(dummy_input)
        self.model.train()
        
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {dummy_output.shape}")
        print(f"  Output sum: {dummy_output.sum(dim=1).item():.4f}")
        
        if dummy_output.shape[1] != 21:
            raise ValueError(
                f"ERROR: Model outputs {dummy_output.shape[1]} classes after loading checkpoint!\n"
                f"  Expected: 21 classes (Blood removed)\n"
                f"  This means the checkpoint loading overwrote the model architecture.\n"
                f"  Check that config['model']['num_classes'] = 21"
            )
        
        output_sum = dummy_output.sum(dim=1).item()
        if abs(output_sum - 1.0) < 0.01:
            raise ValueError(
                f"ERROR: After checkpoint loading, outputs sum to 1.0 (normalized)!\n"
                f"  This means normalize_output got reset to True.\n"
                f"  The checkpoint may have overwritten this parameter."
            )
        
        print(f"  ✓ Model correctly outputs 21 classes")
        print(f"  ✓ Outputs remain absolute (sum={output_sum:.4f} != 1.0)")
        print(f"  ✓ Ready for Stage 2 training")


def main():
    parser = argparse.ArgumentParser(description='Stage 2 Blood-Masked Training')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config_stage2_bloodmasked.yaml')
    
    args = parser.parse_args()
    
    # CRITICAL: Check that model_deconvolution_absolute.py exists
    script_dir = Path(__file__).parent.absolute()
    model_file = script_dir / 'model_deconvolution_absolute.py'
    
    if not model_file.exists():
        print("\n" + "="*80)
        print("ERROR: Required file missing!")
        print("="*80)
        print(f"\nCannot find: {model_file}")
        print(f"\nThis file is required for Stage 2 training with absolute labels.")
        print(f"\nTo fix:")
        print(f"  cd {script_dir}")
        print(f"  cp /mnt/user-data/outputs/model_deconvolution_absolute.py .")
        print(f"\nOr if using a different location:")
        print(f"  cp <path_to>/model_deconvolution_absolute.py {script_dir}/")
        print("="*80)
        sys.exit(1)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*80)
    print("STAGE 2: BLOOD-MASKED TISSUE DECONVOLUTION TRAINING")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Output: {config['output']['save_dir']}")
    print(f"Model file: {model_file} ✓")
    print()
    
    # Initialize trainer
    trainer = Stage2Trainer(config)
    
    # Build components (model must be built before loading checkpoint)
    trainer.build_model()
    
    # Load Phase 3 checkpoint (after model is built)
    trainer.load_pretrained_checkpoint()
    
    # Build remaining components
    trainer.build_dataloaders()
    trainer.build_optimizer()
    
    # Train
    print("\n" + "="*80)
    print("Starting Stage 2 Training")
    print("="*80)
    print(f"Model predicts: {config['model']['num_classes']} tissues (Blood masked)")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print()
    
    trainer.train()
    
    print("\n" + "="*80)
    print("Stage 2 Training Complete!")
    print("="*80)
    print(f"\nBest checkpoint saved at:")
    print(f"  {trainer.checkpoint_dir / 'checkpoint_best.pt'}")


if __name__ == '__main__':
    main()
