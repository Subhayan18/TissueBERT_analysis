#!/usr/bin/env python3
"""
TissueBERT Mixture Deconvolution Training Script with Sparsity Regularization
===============================================================================

Train model to predict tissue proportions in mixed samples using two-stage
architecture with presence detection and sparsity constraints.

Progressive training strategy:
- Phase 1: 2-tissue mixtures (simple binary)
- Phase 2 SPARSE: 3-5 tissue mixtures with sparsity regularization
- Phase 3: Realistic cfDNA mixtures (blood-dominant)

Each phase fine-tunes from previous checkpoint.

Key improvements over original:
- Two-stage architecture (presence detection + proportion estimation)
- Sparsity regularization (eliminates spurious predictions)
- Enhanced metrics (presence accuracy, sparsity loss)

Author: Mixture Deconvolution Project
Date: December 2024
"""

import os
import sys
import time
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json

# Import components - MODIFIED FOR SPARSITY
from model_deconvolution_sparse import (
    TissueBERTDeconvolution,
    load_pretrained_model,
    mixture_loss_with_sparsity,
    compute_true_presence
)
from dataloader_mixture import create_mixture_dataloaders


class DeconvolutionTrainer:
    """Trainer for sparse mixture deconvolution model"""

    def __init__(self, config):
        self.config = config
        self.start_time = time.time()

        # Setup
        self.setup_directories()
        self.setup_device()
        self.setup_logging()
        self.save_config()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_mae = float('inf')

    def setup_directories(self):
        """Create output directories"""
        self.checkpoint_dir = Path(self.config['output']['save_dir']) / 'checkpoints'
        self.log_dir = Path(self.config['output']['save_dir']) / 'logs'
        self.results_dir = Path(self.config['output']['save_dir']) / 'results'

        for dir_path in [self.checkpoint_dir, self.log_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f"\nDirectories:")
        print(f"  Checkpoints: {self.checkpoint_dir}")
        print(f"  Logs: {self.log_dir}")
        print(f"  Results: {self.results_dir}")

    def setup_device(self):
        """Setup device and print GPU info"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nDevice: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Memory: {total_mem:.1f} GB")
        else:
            print("WARNING: Running on CPU (training will be slow)")

    def setup_logging(self):
        """Setup TensorBoard and CSV logging"""
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.log_dir / 'tensorboard')

        # CSV log - UPDATED HEADER FOR SPARSE METRICS
        self.csv_log = self.log_dir / 'training_log.csv'
        if not self.csv_log.exists():
            with open(self.csv_log, 'w') as f:
                f.write('epoch,step,train_loss,train_mae,val_loss,val_mae,presence_acc,lr,time_hrs,gpu_gb\n')

    def save_config(self):
        """Save configuration"""
        config_path = self.log_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        print(f"\n✓ Saved config to {config_path}")

    def build_model(self):
        """Build or load model - MODIFIED FOR TWO-STAGE"""
        print("\n" + "="*80)
        print("Building Model")
        print("="*80)

        model_config = self.config['model']
        
        # Check if we should load from pretrained or previous phase checkpoint
        if 'pretrained_checkpoint' in model_config and model_config['pretrained_checkpoint']:
            checkpoint_path = model_config['pretrained_checkpoint']
            print(f"\nLoading from checkpoint: {checkpoint_path}")
            
            # MODIFIED: Pass two-stage flag
            self.model = load_pretrained_model(
                checkpoint_path,
                device=self.device,
                use_two_stage=model_config.get('use_two_stage', True),
                verbose=True
            )
        else:
            print("\nInitializing new model (no pretrained weights)")
            self.model = TissueBERTDeconvolution(
                vocab_size=model_config.get('vocab_size', 69),
                hidden_size=model_config['hidden_size'],
                num_hidden_layers=model_config.get('num_hidden_layers', 3),
                num_attention_heads=model_config.get('num_attention_heads', 4),
                intermediate_size=model_config.get('intermediate_size', 2048),
                max_position_embeddings=model_config.get('max_position_embeddings', 150),
                num_classes=model_config['num_classes'],
                dropout=model_config['dropout'],
                n_regions=model_config['n_regions'],
                # NEW PARAMETERS FOR SPARSITY
                use_two_stage=model_config.get('use_two_stage', True),
                presence_threshold=model_config.get('presence_threshold', 0.5),
                sparsity_regularization=self.config['training'].get('sparsity_regularization', True)
            ).to(self.device)

        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nTrainable parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    def build_dataloaders(self):
        """Build data loaders"""
        print("\n" + "="*80)
        print("Building DataLoaders")
        print("="*80)

        data_config = self.config['data']
        training_config = self.config['training']

        self.train_loader, self.val_loader, self.test_loader = create_mixture_dataloaders(
            hdf5_path=data_config['hdf5_path'],
            metadata_csv=data_config['metadata_csv'],
            validation_h5=data_config['validation_h5'],
            test_h5=data_config['test_h5'],
            phase=training_config['phase'],
            batch_size=training_config['batch_size'],
            n_mixtures_per_epoch=training_config['mixtures_per_epoch'],
            pure_sample_ratio=training_config['pure_sample_ratio'],
            num_workers=training_config['num_workers'],
            seed=self.config['random_seed']
        )

    def build_optimizer(self):
        """Build optimizer and scheduler"""
        print("\n" + "="*80)
        print("Building Optimizer")
        print("="*80)

        training_config = self.config['training']
        opt_config = self.config['optimizer']

        # Optimizer
        if opt_config['type'] == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                betas=opt_config['betas'],
                eps=opt_config['eps'],
                weight_decay=training_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['type']}")

        print(f"Optimizer: {opt_config['type']}")
        print(f"  Learning rate: {training_config['learning_rate']}")
        print(f"  Weight decay: {training_config['weight_decay']}")

        # Learning rate scheduler
        scheduler_config = self.config['scheduler']
        total_steps = len(self.train_loader) * training_config['num_epochs']
        warmup_steps = int(total_steps * training_config['warmup_ratio'])

        if scheduler_config['type'] == 'cosine':
            # Warmup + Cosine decay
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    return 0.5 * (1 + np.cos(np.pi * progress))
            
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lr_lambda
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_config['type']}")

        print(f"Scheduler: {scheduler_config['type']}")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")

    def train_epoch(self, epoch):
        """Train for one epoch - MODIFIED FOR TWO-STAGE SPARSE"""
        self.model.train()
        
        train_loss = 0.0
        train_mae = 0.0
        train_presence_acc = 0.0
        n_batches = len(self.train_loader)
        
        # Gradient accumulation
        accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        
        # Loss config
        loss_config = self.config['loss']
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            methylation = batch['methylation'].to(self.device)
            true_proportions = batch['proportions'].to(self.device)
            
            # MODIFIED: Two-stage forward pass
            if self.model.use_two_stage:
                pred_proportions, presence_probs = self.model(methylation, return_presence=True)
                true_presence = compute_true_presence(
                    true_proportions,
                    threshold=loss_config.get('presence_threshold', 0.01)
                )
            else:
                pred_proportions = self.model(methylation)
                presence_probs = None
                true_presence = None
            
            # MODIFIED: Sparse loss with presence and sparsity terms
            loss, loss_dict = mixture_loss_with_sparsity(
                pred_proportions, true_proportions,
                presence_probs, true_presence,
                mse_weight=loss_config.get('mse_weight', 1.0),
                presence_weight=loss_config.get('presence_weight', 1.0),
                sparsity_weight=loss_config.get('sparsity_weight', 0.01)
            )
            
            loss = loss / accumulation_steps  # Scale for gradient accumulation
            
            # Backward pass
            loss.backward()
            
            # Update weights (every accumulation_steps)
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Compute MAE
            mae = torch.abs(pred_proportions - true_proportions).mean()
            
            # NEW: Compute presence accuracy
            if presence_probs is not None:
                presence_acc = ((presence_probs > 0.5).float() == true_presence).float().mean()
                train_presence_acc += presence_acc.item()
            else:
                presence_acc = torch.tensor(0.0)
            
            # Accumulate metrics
            train_loss += loss.item() * accumulation_steps
            train_mae += mae.item()
            
            # Update progress bar
            postfix = {
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'mae': f'{mae.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            }
            if presence_probs is not None:
                postfix['pres_acc'] = f'{presence_acc.item():.3f}'
            pbar.set_postfix(postfix)
            
            # Log to TensorBoard - MODIFIED FOR SPARSE METRICS
            if self.global_step % self.config['output']['log_every_n_steps'] == 0:
                self.writer.add_scalar('train/total_loss', loss.item() * accumulation_steps, self.global_step)
                self.writer.add_scalar('train/mae', mae.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)
                
                # NEW: Log individual loss components
                if 'mse_loss' in loss_dict:
                    self.writer.add_scalar('train/mse_loss', loss_dict['mse_loss'], self.global_step)
                if 'presence_loss' in loss_dict:
                    self.writer.add_scalar('train/presence_loss', loss_dict['presence_loss'], self.global_step)
                if 'sparsity_loss' in loss_dict:
                    self.writer.add_scalar('train/sparsity_loss', loss_dict['sparsity_loss'], self.global_step)
                
                if presence_probs is not None:
                    self.writer.add_scalar('train/presence_accuracy', presence_acc.item(), self.global_step)
        
        # Average metrics
        train_loss /= n_batches
        train_mae /= n_batches
        train_presence_acc /= n_batches if presence_probs is not None else 1.0
        
        return train_loss, train_mae, train_presence_acc

    @torch.no_grad()
    def validate(self, loader):
        """Validate on given loader - MODIFIED FOR TWO-STAGE SPARSE"""
        self.model.eval()
        
        val_loss = 0.0
        val_mae = 0.0
        val_presence_acc = 0.0
        all_preds = []
        all_trues = []
        
        # Loss config
        loss_config = self.config['loss']
        
        for batch in tqdm(loader, desc="Validating"):
            methylation = batch['methylation'].to(self.device)
            true_proportions = batch['proportions'].to(self.device)
            
            # MODIFIED: Two-stage forward pass
            if self.model.use_two_stage:
                pred_proportions, presence_probs = self.model(methylation, return_presence=True)
                true_presence = compute_true_presence(
                    true_proportions,
                    threshold=loss_config.get('presence_threshold', 0.01)
                )
            else:
                pred_proportions = self.model(methylation)
                presence_probs = None
                true_presence = None
            
            # MODIFIED: Sparse loss
            loss, loss_dict = mixture_loss_with_sparsity(
                pred_proportions, true_proportions,
                presence_probs, true_presence,
                mse_weight=loss_config.get('mse_weight', 1.0),
                presence_weight=loss_config.get('presence_weight', 1.0),
                sparsity_weight=loss_config.get('sparsity_weight', 0.01)
            )
            
            # Compute metrics
            mae = torch.abs(pred_proportions - true_proportions).mean()
            
            # NEW: Presence accuracy
            if presence_probs is not None:
                presence_acc = ((presence_probs > 0.5).float() == true_presence).float().mean()
                val_presence_acc += presence_acc.item()
            
            val_loss += loss.item()
            val_mae += mae.item()
            
            # Store predictions
            all_preds.append(pred_proportions.cpu())
            all_trues.append(true_proportions.cpu())
        
        # Average metrics
        val_loss /= len(loader)
        val_mae /= len(loader)
        val_presence_acc /= len(loader) if presence_probs is not None else 1.0
        
        # Concatenate all predictions
        all_preds = torch.cat(all_preds, dim=0)
        all_trues = torch.cat(all_trues, dim=0)
        
        # Additional metrics
        per_tissue_mae = torch.abs(all_preds - all_trues).mean(dim=0)
        
        return val_loss, val_mae, val_presence_acc, per_tissue_mae, all_preds, all_trues

    def save_checkpoint(self, epoch, is_best=False, is_last=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_mae': self.best_val_mae,
            'config': self.config
        }
        
        if is_best:
            path = self.checkpoint_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, path)
            print(f"\n✓ Saved best checkpoint to {path}")
        
        if is_last:
            path = self.checkpoint_dir / 'checkpoint_last.pt'
            torch.save(checkpoint, path)
            print(f"✓ Saved last checkpoint to {path}")
        
        # Save periodic checkpoints
        if epoch % 5 == 0:
            path = self.checkpoint_dir / f'checkpoint_epoch{epoch}.pt'
            torch.save(checkpoint, path)
            print(f"✓ Saved epoch {epoch} checkpoint to {path}")

    def log_epoch(self, epoch, train_loss, train_mae, train_pres_acc, 
                  val_loss, val_mae, val_pres_acc):
        """Log epoch metrics - MODIFIED FOR SPARSE METRICS"""
        # Time
        elapsed = (time.time() - self.start_time) / 3600
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / 1e9
        else:
            gpu_mem = 0.0
        
        # Current learning rate
        lr = self.scheduler.get_last_lr()[0]
        
        # TensorBoard
        self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
        self.writer.add_scalar('epoch/train_mae', train_mae, epoch)
        self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
        self.writer.add_scalar('epoch/val_mae', val_mae, epoch)
        
        # NEW: Presence accuracy
        if self.model.use_two_stage:
            self.writer.add_scalar('epoch/train_presence_acc', train_pres_acc, epoch)
            self.writer.add_scalar('epoch/val_presence_acc', val_pres_acc, epoch)
        
        # CSV - UPDATED FORMAT
        with open(self.csv_log, 'a') as f:
            f.write(f'{epoch},{self.global_step},{train_loss:.6f},{train_mae:.6f},'
                   f'{val_loss:.6f},{val_mae:.6f},{val_pres_acc:.6f},'
                   f'{lr:.2e},{elapsed:.2f},{gpu_mem:.2f}\n')
        
        # Console
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train - Loss: {train_loss:.6f}, MAE: {train_mae:.6f}", end='')
        if self.model.use_two_stage:
            print(f", Presence Acc: {train_pres_acc:.6f}")
        else:
            print()
        
        print(f"  Val   - Loss: {val_loss:.6f}, MAE: {val_mae:.6f}", end='')
        if self.model.use_two_stage:
            print(f", Presence Acc: {val_pres_acc:.6f}")
        else:
            print()
        
        print(f"  Time: {elapsed:.2f} hrs, GPU: {gpu_mem:.2f} GB, LR: {lr:.2e}")

    def train(self):
        """Main training loop"""
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
        
        training_config = self.config['training']
        num_epochs = training_config['num_epochs']
        val_freq = training_config.get('validation_frequency', 5)
        
        print(f"\nConfiguration:") 
        print(f"  Phase: {training_config['phase']}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {training_config['batch_size']}")
        print(f"  Validation frequency: every {val_freq} epochs")
        print(f"  Mixtures per epoch: {training_config['mixtures_per_epoch']}")
        
        # NEW: Print sparsity config
        if self.model.use_two_stage:
            print(f"\n  SPARSITY ENABLED:")
            print(f"    Two-stage architecture: Yes")
            print(f"    Presence threshold (inference): {self.model.presence_threshold}")
            print(f"    Presence threshold (labels): {self.config['loss'].get('presence_threshold', 0.01)}")
            print(f"    Sparsity weight: {self.config['loss'].get('sparsity_weight', 0.01)}")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*80}")
            
            # Train
            train_loss, train_mae, train_pres_acc = self.train_epoch(epoch)
            
            # Validate
            if epoch % val_freq == 0 or epoch == num_epochs:
                val_loss, val_mae, val_pres_acc, per_tissue_mae, preds, trues = \
                    self.validate(self.val_loader)
                
                # Log
                self.log_epoch(epoch, train_loss, train_mae, train_pres_acc,
                              val_loss, val_mae, val_pres_acc)
                
                # Check if best
                is_best = val_mae < self.best_val_mae
                if is_best:
                    self.best_val_mae = val_mae
                    self.best_val_loss = val_loss
                    print(f"\n🎉 New best model! MAE: {val_mae:.6f}")
                
                # Save checkpoints
                self.save_checkpoint(epoch, is_best=is_best, is_last=True)
            else:
                # Just log training metrics
                print(f"\nEpoch {epoch}: Train Loss={train_loss:.6f}, MAE={train_mae:.6f}", end='')
                if self.model.use_two_stage:
                    print(f", Presence Acc={train_pres_acc:.6f}")
                else:
                    print()
                self.save_checkpoint(epoch, is_best=False, is_last=True)
        
        # Final evaluation on test set
        print("\n" + "="*80)
        print("FINAL EVALUATION ON TEST SET")
        print("="*80)
        
        test_loss, test_mae, test_pres_acc, per_tissue_mae, preds, trues = \
            self.validate(self.test_loader)
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.6f}")
        print(f"  MAE: {test_mae:.6f}")
        if self.model.use_two_stage:
            print(f"  Presence Accuracy: {test_pres_acc:.6f}")
        
        # Save test results
        results = {
            'test_loss': float(test_loss),
            'test_mae': float(test_mae),
            'test_presence_acc': float(test_pres_acc) if self.model.use_two_stage else None,
            'per_tissue_mae': per_tissue_mae.tolist()
        }
        
        results_path = self.results_dir / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Saved test results to {results_path}")
        
        # Training complete
        total_time = (time.time() - self.start_time) / 3600
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print(f"Total time: {total_time:.2f} hours")
        print(f"Best validation MAE: {self.best_val_mae:.6f}")
        print(f"Final test MAE: {test_mae:.6f}")
        print("="*80)

    def run(self):
        """Run complete training pipeline"""
        try:
            # Build components
            self.build_model()
            self.build_dataloaders()
            self.build_optimizer()
            
            # Train
            self.train()
            
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            print("Saving checkpoint...")
            self.save_checkpoint(self.current_epoch, is_best=False, is_last=True)
            print("✓ Checkpoint saved")
            
        except Exception as e:
            print(f"\n\nERROR: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            # Close TensorBoard writer
            self.writer.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train TissueBERT Deconvolution with Sparsity')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    
    args = parser.parse_args()
    
    # Load config
    print("="*80)
    print("TISSUEBERT MIXTURE DECONVOLUTION TRAINING (SPARSE)")
    print("="*80)
    print(f"\nConfig file: {args.config}")
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    seed = config.get('random_seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"Random seed: {seed}")
    
    # Create trainer and run
    trainer = DeconvolutionTrainer(config)
    trainer.run()


if __name__ == '__main__':
    main()
