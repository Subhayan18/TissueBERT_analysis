#!/usr/bin/env python3
"""
TissueBERT Training Script - FILE-LEVEL VERSION
Multi-class tissue classification from methylation patterns

Modified to use file-level classification (all regions from a file)
to match logistic regression approach.
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

# Import file-level dataloader and CNN+Attention model
from dataloader_filelevel import create_filelevel_dataloaders
from model_cnn_attention import CNNAttentionTissueBERT as TissueBERT
from utils import (
    AverageMeter,
    compute_metrics,
    save_checkpoint,
    load_checkpoint,
    plot_confusion_matrix,
    save_training_summary
)


class Trainer:
    """Clean, robust trainer for TissueBERT"""

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
        self.best_val_acc = 0.0
        self.resume_count = 0

    def setup_directories(self):
        """Create directories"""
        self.checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        self.log_dir = Path(self.config['paths']['log_dir'])
        self.results_dir = Path(self.config['paths']['results_dir'])

        for dir_path in [self.checkpoint_dir, self.log_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f"Directories:")
        print(f"  Checkpoints: {self.checkpoint_dir}")
        print(f"  Logs: {self.log_dir}")
        print(f"  Results: {self.results_dir}")

    def setup_device(self):
        """Setup device"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nDevice: {self.device}")
        if torch.cuda.is_available():
            # CRITICAL: Clear any leftover GPU memory from previous runs
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

    def setup_logging(self):
        """Setup TensorBoard and CSV logging"""
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.log_dir / 'tensorboard')

        # CSV log
        self.csv_log = self.log_dir / 'training_log.csv'
        if not self.csv_log.exists():
            with open(self.csv_log, 'w') as f:
                f.write('epoch,step,train_loss,train_acc,val_loss,val_acc,lr,time_hrs,gpu_gb\n')

    def save_config(self):
        """Save config"""
        with open(self.log_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)

    def build_model(self):
        """Build model"""
        print("\n" + "="*80)
        print("Building Model")
        print("="*80)

        cfg = self.config['model']
        self.model = TissueBERT(
            vocab_size=cfg['vocab_size'],
            hidden_size=cfg['hidden_size'],
            num_hidden_layers=cfg['num_layers'],
            num_attention_heads=cfg['num_attention_heads'],
            intermediate_size=cfg['intermediate_size'],
            max_position_embeddings=cfg['max_seq_length'],
            num_classes=cfg['num_classes'],
            dropout=cfg['dropout'],
            region_embed_dim=cfg.get('region_embed_dim', 256),
            region_chunk_size=cfg.get('region_chunk_size', 512)
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Parameters: {total_params:,}")
        print(f"Size: {total_params * 4 / 1e6:.2f} MB")

        return self.model

    def build_dataloaders(self):
        """Build dataloaders"""
        print("\n" + "="*80)
        print("Building DataLoaders")
        print("="*80)

        data_cfg = self.config['data']
        train_cfg = self.config['training']

        train_loader, val_loader, test_loader = create_filelevel_dataloaders(
            hdf5_path=data_cfg['hdf5_path'],
            train_csv=data_cfg['train_csv'],
            val_csv=data_cfg['val_csv'],
            test_csv=data_cfg['test_csv'],
            batch_size=8,  # Smaller batch size - each sample is entire file
            num_workers=train_cfg['num_workers'],
            pin_memory=True
        )

        print(f"Batches: train={len(train_loader):,}, val={len(val_loader):,}, test={len(test_loader):,}")

        return train_loader, val_loader, test_loader

    def build_optimizer(self):
        """Build optimizer and scheduler"""
        print("\n" + "="*80)
        print("Building Optimizer & Scheduler")
        print("="*80)

        opt_cfg = self.config['optimizer']
        train_cfg = self.config['training']

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt_cfg['learning_rate'],
            betas=(opt_cfg['beta1'], opt_cfg['beta2']),
            eps=opt_cfg['eps'],
            weight_decay=opt_cfg['weight_decay']
        )

        # Calculate scheduler steps CORRECTLY
        max_steps = train_cfg.get('max_steps_per_epoch')
        if max_steps is None:
            batches_per_epoch = len(self.train_loader)
        else:
            batches_per_epoch = min(max_steps, len(self.train_loader))

        grad_accum = train_cfg['gradient_accumulation_steps']
        optimizer_steps_per_epoch = batches_per_epoch // grad_accum
        total_optimizer_steps = optimizer_steps_per_epoch * train_cfg['num_epochs']

        print(f"\nScheduler Calculation:")
        print(f"  Batches/epoch: {batches_per_epoch:,}")
        print(f"  Grad accum: {grad_accum}")
        print(f"  Optimizer steps/epoch: {optimizer_steps_per_epoch:,}")
        print(f"  Total optimizer steps: {total_optimizer_steps:,}")

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=opt_cfg['learning_rate'],
            total_steps=total_optimizer_steps,
            pct_start=opt_cfg['warmup_ratio'],
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1e4
        )

        print(f"\nOptimizer: AdamW (lr={opt_cfg['learning_rate']:.2e}, wd={opt_cfg['weight_decay']})")
        print(f"Scheduler: OneCycleLR (warmup={int(total_optimizer_steps * opt_cfg['warmup_ratio']):,} steps)")

        return self.optimizer, self.scheduler

    def build_criterion(self):
        """Build loss function"""
        loss_cfg = self.config['loss']
        self.criterion = nn.CrossEntropyLoss(
            reduction='mean',
            label_smoothing=loss_cfg['label_smoothing']
        )
        print(f"\nLoss: CrossEntropyLoss (smoothing={loss_cfg['label_smoothing']})")
        return self.criterion

    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        max_steps = self.config['training'].get('max_steps_per_epoch')
        if max_steps is None:
            max_steps = len(self.train_loader)

        grad_accum = self.config['training']['gradient_accumulation_steps']
        self.optimizer.zero_grad()

        pbar = tqdm(
            enumerate(self.train_loader),
            total=max_steps,
            desc=f"Epoch {epoch}/{self.config['training']['num_epochs']}"
        )

        for batch_idx, batch in pbar:
            if batch_idx >= max_steps:
                break

            # Forward
            dna = batch['dna_tokens'].to(self.device, non_blocking=True)
            meth = batch['methylation'].to(self.device, non_blocking=True)
            targets = batch['tissue_label'].to(self.device, non_blocking=True)

            outputs = self.model(dna, meth)
            loss = self.criterion(outputs, targets) / grad_accum

            # Backward
            loss.backward()

            # Update every grad_accum steps
            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_grad_norm']
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Metrics
            preds = outputs.argmax(dim=1)
            acc = (preds == targets).float().mean().item()

            loss_meter.update(loss.item() * grad_accum, targets.size(0))
            acc_meter.update(acc, targets.size(0))
            
            # CRITICAL: Clear GPU cache to prevent memory accumulation
            # Especially important with chunked processing
            del dna, meth, targets, outputs, loss, preds
            if batch_idx % 10 == 0:  # Clear cache every 10 batches
                torch.cuda.empty_cache()

            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })

            # Log to TensorBoard
            if self.global_step % self.config['logging']['log_interval'] == 0:
                self.writer.add_scalar('train/loss', loss_meter.avg, self.global_step)
                self.writer.add_scalar('train/acc', acc_meter.avg, self.global_step)
                self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)

        return loss_meter.avg, acc_meter.avg

    @torch.no_grad()
    def validate(self, loader, split='val'):
        """Validate"""
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        all_preds = []
        all_targets = []
        all_probs = []

        max_val_steps = self.config['training'].get('max_val_steps')
        if max_val_steps is None:
            max_val_steps = len(loader)

        pbar = tqdm(loader, total=max_val_steps, desc=f"Validating ({split})")

        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= max_val_steps:
                break

            dna = batch['dna_tokens'].to(self.device, non_blocking=True)
            meth = batch['methylation'].to(self.device, non_blocking=True)
            targets = batch['tissue_label'].to(self.device, non_blocking=True)

            outputs = self.model(dna, meth)
            loss = self.criterion(outputs, targets)

            preds = outputs.argmax(dim=1)
            acc = (preds == targets).float().mean().item()
            probs = F.softmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            loss_meter.update(loss.item(), targets.size(0))
            acc_meter.update(acc, targets.size(0))
            
            # Clear GPU memory
            del dna, meth, targets, outputs, preds, probs, loss
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 'acc': f'{acc_meter.avg:.4f}'})

        # Compute metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        metrics = compute_metrics(
            predictions=all_preds,
            targets=all_targets,
            probs=all_probs,
            num_classes=self.config['model']['num_classes']
        )

        return loss_meter.avg, acc_meter.avg, metrics

    def log_epoch_results(self, epoch, train_loss, train_acc, val_loss, val_acc, val_metrics):
        """Log epoch results"""
        elapsed = (time.time() - self.start_time) / 3600
        gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        lr = self.scheduler.get_last_lr()[0]

        # TensorBoard
        self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
        self.writer.add_scalar('epoch/train_acc', train_acc, epoch)
        self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
        self.writer.add_scalar('epoch/val_acc', val_acc, epoch)
        self.writer.add_scalar('epoch/lr', lr, epoch)

        # CSV
        with open(self.csv_log, 'a') as f:
            f.write(f"{epoch},{self.global_step},{train_loss:.6f},{train_acc:.6f},")
            f.write(f"{val_loss:.6f},{val_acc:.6f},{lr:.8f},{elapsed:.2f},{gpu_mem:.2f}\n")

        # Console
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{self.config['training']['num_epochs']} Summary")
        print(f"{'='*80}")
        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        print(f"Time: {elapsed:.2f}h | GPU: {gpu_mem:.2f} GB | LR: {lr:.2e}")
        print(f"{'='*80}\n")

    def save_checkpoint_if_best(self, epoch, val_loss, val_acc, val_metrics):
        """Save checkpoint"""
        is_best_loss = val_loss < self.best_val_loss
        is_best_acc = val_acc > self.best_val_acc

        if is_best_loss:
            self.best_val_loss = val_loss
        if is_best_acc:
            self.best_val_acc = val_acc

        checkpoint_data = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'resume_count': self.resume_count,
            'config': self.config,
            'val_metrics': val_metrics
        }

        save_checkpoint(checkpoint_data, self.checkpoint_dir / 'checkpoint_last.pt')

        if is_best_loss:
            save_checkpoint(checkpoint_data, self.checkpoint_dir / 'checkpoint_best_loss.pt')
            print(f"✓ New best val loss: {val_loss:.4f}")

        if is_best_acc:
            save_checkpoint(checkpoint_data, self.checkpoint_dir / 'checkpoint_best_acc.pt')
            print(f"✓ New best val acc: {val_acc:.4f}")

        if epoch % self.config['training']['save_interval'] == 0:
            save_checkpoint(checkpoint_data, self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pt')

    def train(self):
        """Main training loop"""
        print("\n" + "="*80)
        print("Starting Training - FILE-LEVEL CLASSIFICATION")
        print("="*80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Build components
        self.model = self.build_model()
        self.train_loader, self.val_loader, self.test_loader = self.build_dataloaders()
        self.optimizer, self.scheduler = self.build_optimizer()
        self.criterion = self.build_criterion()

        # Training loop
        num_epochs = self.config['training']['num_epochs']
        val_freq = self.config['training'].get('validation_frequency', 1)

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch

            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            if (epoch % val_freq == 0) or (epoch == num_epochs):
                val_loss, val_acc, val_metrics = self.validate(self.val_loader, 'val')
                self.log_epoch_results(epoch, train_loss, train_acc, val_loss, val_acc, val_metrics)
                self.save_checkpoint_if_best(epoch, val_loss, val_acc, val_metrics)
            else:
                print(f"\nEpoch {epoch}/{num_epochs}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f}")

        # Final test
        print("\n" + "="*80)
        print("Final Test Evaluation")
        print("="*80)

        best_ckpt = self.checkpoint_dir / 'checkpoint_best_acc.pt'
        if best_ckpt.exists():
            checkpoint = torch.load(best_ckpt, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model (val acc={checkpoint['best_val_acc']:.4f})")

        test_loss, test_acc, test_metrics = self.validate(self.test_loader, 'test')

        print(f"\nTest Results:")
        print(f"  Accuracy: {test_acc:.4f}")

        self.writer.close()

        total_time = (time.time() - self.start_time) / 3600
        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"Time: {total_time:.2f} hours")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Train TissueBERT - File Level')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    args = parser.parse_args()

    # CRITICAL: Clear GPU memory before starting
    # This handles any leftover memory from crashed/killed jobs
    if torch.cuda.is_available():
        print("="*80)
        print("GPU Memory Cleanup")
        print("="*80)
        print(f"Memory before cleanup: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")
        print(f"                       {torch.cuda.memory_reserved(0) / 1e9:.2f} GB reserved")
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        print(f"Memory after cleanup:  {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")
        print(f"                       {torch.cuda.memory_reserved(0) / 1e9:.2f} GB reserved")
        print("="*80 + "\n")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
