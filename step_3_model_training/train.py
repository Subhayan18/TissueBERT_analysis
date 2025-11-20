#!/usr/bin/env python3
"""
TissueBERT Training Script
Multi-class tissue classification from methylation patterns

Author: Step 2.3 Training Setup
Date: 2025-11-20
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

# Add the step2.2 directory to path for dataloader import
sys.path.append('/home/chattopa/data_storage/TissueBERT_analysis/step_2_model_architecture/step2.2')
from dataset_dataloader import create_dataloaders

# Import model and utils (will be in same directory)
from model import TissueBERT
from utils import (
    AverageMeter, 
    ProgressLogger, 
    compute_metrics,
    save_checkpoint,
    load_checkpoint,
    plot_confusion_matrix,
    save_training_summary
)


class Trainer:
    """Main training class for TissueBERT"""
    
    def __init__(self, config):
        self.config = config
        self.start_time = time.time()
        
        # Setup directories
        self.setup_directories()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Initialize tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.resume_count = 0
        
        # Setup logging
        self.setup_logging()
        
        # Save config
        self.save_config()
        
    def setup_directories(self):
        """Create all necessary directories"""
        self.checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        self.log_dir = Path(self.config['paths']['log_dir'])
        self.results_dir = Path(self.config['paths']['results_dir'])
        
        for dir_path in [self.checkpoint_dir, self.log_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"Checkpoints: {self.checkpoint_dir}")
        print(f"Logs: {self.log_dir}")
        print(f"Results: {self.results_dir}")
        
    def setup_logging(self):
        """Setup TensorBoard and CSV logging"""
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.log_dir / 'tensorboard')
        
        # CSV logging
        self.csv_log = self.log_dir / 'training_log.csv'
        if not self.csv_log.exists():
            with open(self.csv_log, 'w') as f:
                f.write('epoch,step,train_loss,train_acc,val_loss,val_acc,')
                f.write('learning_rate,time_elapsed,gpu_memory\n')
        
        # Per-tissue metrics log
        self.tissue_log = self.log_dir / 'per_tissue_metrics.csv'
        if not self.tissue_log.exists():
            with open(self.tissue_log, 'w') as f:
                header = 'epoch,tissue_id'
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    header += f',{metric}'
                f.write(header + '\n')
                
        # Training summary log
        self.summary_log = self.log_dir / 'training_summary.txt'
        
    def save_config(self):
        """Save configuration to log directory"""
        config_path = self.log_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        print(f"Config saved: {config_path}")
        
    def build_model(self):
        """Build and initialize the model"""
        print("\n" + "="*80)
        print("Building Model")
        print("="*80)
        
        model_config = self.config['model']
        self.model = TissueBERT(
            vocab_size=model_config['vocab_size'],
            hidden_size=model_config['hidden_size'],
            num_hidden_layers=model_config['num_layers'],
            num_attention_heads=model_config['num_attention_heads'],
            intermediate_size=model_config['intermediate_size'],
            max_position_embeddings=model_config['max_seq_length'],
            num_classes=model_config['num_classes'],
            dropout=model_config['dropout']
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {total_params * 4 / 1e6:.2f} MB (fp32)")
        
        return self.model
    
    def build_dataloaders(self):
        """Build train, validation, and test dataloaders"""
        print("\n" + "="*80)
        print("Building DataLoaders")
        print("="*80)
        
        data_config = self.config['data']
        train_config = self.config['training']
        
        train_loader, val_loader, test_loader = create_dataloaders(
            hdf5_path=data_config['hdf5_path'],
            train_csv=data_config['train_csv'],
            val_csv=data_config['val_csv'],
            test_csv=data_config['test_csv'],
            batch_size=train_config['batch_size'],
            num_workers=train_config['num_workers'],
            cpg_dropout_rate=train_config['cpg_dropout_rate'],
            tissue_balanced=True,
            pin_memory=True
        )
        
        print(f"Train batches per epoch: {len(train_loader):,}")
        print(f"Validation batches: {len(val_loader):,}")
        print(f"Test batches: {len(test_loader):,}")
        
        return train_loader, val_loader, test_loader
    
    def build_optimizer(self):
        """Build optimizer and scheduler"""
        print("\n" + "="*80)
        print("Building Optimizer & Scheduler")
        print("="*80)
        
        opt_config = self.config['optimizer']
        train_config = self.config['training']
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt_config['learning_rate'],
            betas=(opt_config['beta1'], opt_config['beta2']),
            eps=opt_config['eps'],
            weight_decay=opt_config['weight_decay']
        )
        
        # Calculate total steps
        steps_per_epoch = len(self.train_loader) // train_config['gradient_accumulation_steps']
        total_steps = steps_per_epoch * train_config['num_epochs']
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=opt_config['learning_rate'],
            total_steps=total_steps,
            pct_start=opt_config['warmup_ratio'],
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1e4
        )
        
        print(f"Optimizer: AdamW")
        print(f"Learning rate: {opt_config['learning_rate']}")
        print(f"Weight decay: {opt_config['weight_decay']}")
        print(f"Scheduler: OneCycleLR")
        print(f"Total steps: {total_steps:,}")
        print(f"Steps per epoch: {steps_per_epoch:,}")
        print(f"Warmup steps: {int(total_steps * opt_config['warmup_ratio']):,}")
        
        return self.optimizer, self.scheduler
    
    def build_criterion(self):
        """Build loss function"""
        loss_config = self.config['loss']
        
        self.criterion = nn.CrossEntropyLoss(
            reduction='mean',
            label_smoothing=loss_config['label_smoothing']
        )
        
        print(f"\nLoss: CrossEntropyLoss")
        print(f"Label smoothing: {loss_config['label_smoothing']}")
        
        return self.criterion
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        # Meters for tracking
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        # Progress bar
        pbar = tqdm(
            enumerate(self.train_loader), 
            total=len(self.train_loader),
            desc=f"Epoch {epoch}/{self.config['training']['num_epochs']}"
        )
        
        # Gradient accumulation
        grad_accum_steps = self.config['training']['gradient_accumulation_steps']
        self.optimizer.zero_grad()
        
        for batch_idx, batch in pbar:
            # Move to device
            dna_tokens = batch['dna_tokens'].to(self.device, non_blocking=True)
            methylation = batch['methylation'].to(self.device, non_blocking=True)
            targets = batch['tissue_label'].to(self.device, non_blocking=True)
            
            # Forward pass
            outputs = self.model(dna_tokens, methylation)
            loss = self.criterion(outputs, targets)
            
            # Normalize loss for gradient accumulation
            loss = loss / grad_accum_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every grad_accum_steps
            if (batch_idx + 1) % grad_accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['max_grad_norm']
                )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Increment global step
                self.global_step += 1
            
            # Calculate accuracy
            predictions = outputs.argmax(dim=1)
            accuracy = (predictions == targets).float().mean().item()
            
            # Update meters (denormalize loss)
            loss_meter.update(loss.item() * grad_accum_steps, targets.size(0))
            acc_meter.update(accuracy, targets.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log to TensorBoard every N steps
            if self.global_step % self.config['logging']['log_interval'] == 0:
                self.writer.add_scalar('train/loss_step', loss_meter.avg, self.global_step)
                self.writer.add_scalar('train/acc_step', acc_meter.avg, self.global_step)
                self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)
        
        return loss_meter.avg, acc_meter.avg
    
    @torch.no_grad()
    def validate(self, data_loader, split='val'):
        """Validate the model"""
        self.model.eval()
        
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        all_predictions = []
        all_targets = []
        all_probs = []
        
        pbar = tqdm(data_loader, desc=f"Validating ({split})")
        
        for batch in pbar:
            # Move to device
            dna_tokens = batch['dna_tokens'].to(self.device, non_blocking=True)
            methylation = batch['methylation'].to(self.device, non_blocking=True)
            targets = batch['tissue_label'].to(self.device, non_blocking=True)
            
            # Forward pass
            outputs = self.model(dna_tokens, methylation)
            loss = self.criterion(outputs, targets)
            
            # Calculate accuracy
            predictions = outputs.argmax(dim=1)
            accuracy = (predictions == targets).float().mean().item()
            
            # Update meters
            loss_meter.update(loss.item(), targets.size(0))
            acc_meter.update(accuracy, targets.size(0))
            
            # Store for metrics
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_probs.append(F.softmax(outputs, dim=1).cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.4f}'
            })
        
        # Concatenate all predictions
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)
        
        # Compute detailed metrics
        metrics = compute_metrics(
            all_predictions, 
            all_targets, 
            all_probs,
            num_classes=self.config['model']['num_classes']
        )
        
        return loss_meter.avg, acc_meter.avg, metrics
    
    def log_epoch_results(self, epoch, train_loss, train_acc, val_loss, val_acc, val_metrics):
        """Log results for the epoch"""
        # TensorBoard
        self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
        self.writer.add_scalar('epoch/train_acc', train_acc, epoch)
        self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
        self.writer.add_scalar('epoch/val_acc', val_acc, epoch)
        
        # Log per-tissue metrics
        for tissue_id in range(self.config['model']['num_classes']):
            if tissue_id in val_metrics['per_class_f1']:
                self.writer.add_scalar(
                    f'tissue/f1_tissue_{tissue_id}', 
                    val_metrics['per_class_f1'][tissue_id], 
                    epoch
                )
        
        # CSV logging
        elapsed_time = time.time() - self.start_time
        gpu_memory = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        
        with open(self.csv_log, 'a') as f:
            f.write(f"{epoch},{self.global_step},{train_loss:.6f},{train_acc:.6f},")
            f.write(f"{val_loss:.6f},{val_acc:.6f},")
            f.write(f"{self.scheduler.get_last_lr()[0]:.8f},{elapsed_time:.2f},{gpu_memory:.2f}\n")
        
        # Per-tissue CSV logging
        with open(self.tissue_log, 'a') as f:
            for tissue_id in range(self.config['model']['num_classes']):
                if tissue_id in val_metrics['per_class_accuracy']:
                    f.write(f"{epoch},{tissue_id},")
                    f.write(f"{val_metrics['per_class_accuracy'][tissue_id]:.6f},")
                    f.write(f"{val_metrics['per_class_precision'][tissue_id]:.6f},")
                    f.write(f"{val_metrics['per_class_recall'][tissue_id]:.6f},")
                    f.write(f"{val_metrics['per_class_f1'][tissue_id]:.6f}\n")
        
        # Console logging
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{self.config['training']['num_epochs']} Summary")
        print(f"{'='*80}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"Time elapsed: {elapsed_time/3600:.2f}h")
        print(f"GPU memory: {gpu_memory:.2f} GB")
        print(f"Learning rate: {self.scheduler.get_last_lr()[0]:.2e}")
        print(f"\nTop-3 Accuracy: {val_metrics['top3_acc']:.4f}")
        print(f"Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {val_metrics['weighted_f1']:.4f}")
        
        # Show worst performing tissues
        worst_tissues = sorted(
            val_metrics['per_class_f1'].items(), 
            key=lambda x: x[1]
        )[:5]
        print(f"\nWorst performing tissues (F1):")
        for tissue_id, f1 in worst_tissues:
            print(f"  Tissue {tissue_id}: {f1:.4f}")
        
        print(f"{'='*80}\n")
    
    def save_checkpoint_if_best(self, epoch, val_loss, val_acc, val_metrics):
        """Save checkpoint if validation improved"""
        is_best_loss = val_loss < self.best_val_loss
        is_best_acc = val_acc > self.best_val_acc
        
        # Update best metrics
        if is_best_loss:
            self.best_val_loss = val_loss
        if is_best_acc:
            self.best_val_acc = val_acc
        
        # Save checkpoint
        checkpoint_data = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'resume_count': self.resume_count,
            'val_metrics': val_metrics
        }
        
        # Always save last checkpoint
        last_checkpoint_path = self.checkpoint_dir / 'checkpoint_last.pt'
        save_checkpoint(checkpoint_data, last_checkpoint_path)
        
        # Save best checkpoints
        if is_best_loss:
            best_loss_path = self.checkpoint_dir / 'checkpoint_best_loss.pt'
            save_checkpoint(checkpoint_data, best_loss_path)
            print(f"✓ New best validation loss: {val_loss:.4f}")
        
        if is_best_acc:
            best_acc_path = self.checkpoint_dir / 'checkpoint_best_acc.pt'
            save_checkpoint(checkpoint_data, best_acc_path)
            print(f"✓ New best validation accuracy: {val_acc:.4f}")
        
        # Save periodic checkpoints
        if epoch % self.config['training']['save_interval'] == 0:
            epoch_checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pt'
            save_checkpoint(checkpoint_data, epoch_checkpoint_path)
            print(f"✓ Saved checkpoint: epoch {epoch}")
    
    def load_checkpoint_if_exists(self):
        """Load checkpoint if exists and resume training"""
        checkpoint_path = self.checkpoint_dir / 'checkpoint_last.pt'
        
        if checkpoint_path.exists():
            print(f"\nFound checkpoint: {checkpoint_path}")
            
            # Check resume limit
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.resume_count = checkpoint.get('resume_count', 0)
            
            if self.resume_count >= self.config['training']['max_resume_count']:
                print(f"⚠️  Resume limit reached ({self.resume_count}/{self.config['training']['max_resume_count']})")
                print("Starting fresh training...")
                return False
            
            # Load checkpoint
            print(f"Resuming from epoch {checkpoint['epoch']} (resume #{self.resume_count + 1})")
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_val_loss = checkpoint['best_val_loss']
            self.best_val_acc = checkpoint['best_val_acc']
            self.resume_count += 1
            
            print(f"✓ Loaded checkpoint successfully")
            print(f"  Best val loss: {self.best_val_loss:.4f}")
            print(f"  Best val acc: {self.best_val_acc:.4f}")
            
            return True
        
        return False
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Build components
        self.model = self.build_model()
        self.train_loader, self.val_loader, self.test_loader = self.build_dataloaders()
        self.optimizer, self.scheduler = self.build_optimizer()
        self.criterion = self.build_criterion()
        
        # Try to resume from checkpoint
        resumed = self.load_checkpoint_if_exists()
        
        # Training loop
        start_epoch = self.current_epoch + 1 if resumed else 1
        num_epochs = self.config['training']['num_epochs']
        
        for epoch in range(start_epoch, num_epochs + 1):
            # Train one epoch
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_metrics = self.validate(self.val_loader, split='val')
            
            # Log results
            self.log_epoch_results(epoch, train_loss, train_acc, val_loss, val_acc, val_metrics)
            
            # Save checkpoint
            self.save_checkpoint_if_best(epoch, val_loss, val_acc, val_metrics)
            
            # Save confusion matrix
            if epoch % self.config['logging']['plot_interval'] == 0:
                plot_confusion_matrix(
                    val_metrics['confusion_matrix'],
                    save_path=self.results_dir / f'confusion_epoch_{epoch:03d}.png',
                    num_classes=self.config['model']['num_classes']
                )
        
        # Final evaluation on test set
        print("\n" + "="*80)
        print("Final Evaluation on Test Set")
        print("="*80)
        
        # Load best model
        best_checkpoint = self.checkpoint_dir / 'checkpoint_best_acc.pt'
        if best_checkpoint.exists():
            checkpoint = torch.load(best_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model (val acc: {checkpoint['best_val_acc']:.4f})")
        
        test_loss, test_acc, test_metrics = self.validate(self.test_loader, split='test')
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Top-3 Accuracy: {test_metrics['top3_acc']:.4f}")
        print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
        print(f"  Weighted F1: {test_metrics['weighted_f1']:.4f}")
        
        # Save final results
        final_results = {
            'test_loss': float(test_loss),
            'test_acc': float(test_acc),
            'test_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                           for k, v in test_metrics.items()}
        }
        
        with open(self.results_dir / 'final_results.json', 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # Save final confusion matrix
        plot_confusion_matrix(
            test_metrics['confusion_matrix'],
            save_path=self.results_dir / 'confusion_test_final.png',
            num_classes=self.config['model']['num_classes']
        )
        
        # Generate comprehensive summary
        print("\n" + "="*80)
        print("Generating Training Summary")
        print("="*80)
        
        save_training_summary(
            log_dir=self.log_dir,
            results_dir=self.results_dir,
            config=self.config,
            final_results=final_results
        )
        
        # Close writers
        self.writer.close()
        
        total_time = time.time() - self.start_time
        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"{'='*80}")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Final test accuracy: {test_acc:.4f}")
        print(f"Results saved to: {self.results_dir}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Train TissueBERT model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer and train
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
