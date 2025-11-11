# Phase 3: Model Training

## Goal
Train the DNABERT-S model to classify tissue types from methylation patterns.

## Substeps

### 3.1 Training Configuration
- **Hyperparameters**: Batch size 32, learning rate 4e-4, AdamW optimizer, 50–100 epochs.
- **Features**: Gradient accumulation, mixed precision, early stopping.

### 3.2 Training Pipeline
- **Flow**: Load data → initialize model → train → validate → save best model.

### 3.3 Monitoring & Logging
- **Metrics**: Loss, accuracy, per-tissue F1, confusion matrix, confidence calibration.
- **Tools**: TensorBoard, Weights & Biases, CSV logs.

### 3.4 Training Dynamics
- **Expected Curve**: Rapid loss drop in early epochs, plateau after ~40 epochs.
- **Accuracy Targets**: 85–90% overall, >80% per-tissue F1.

### 3.5 Checkpointing
- **Strategy**: Save every N epochs, best validation model, last checkpoint.
- **Contents**: Model weights, optimizer/scheduler state, training history.

### 3.6 Computational Requirements
- **Single GPU**: ~30–45 min/epoch, ~25–35 hours total.
- **Multi-GPU**: ~6–10 hours total with DDP.

### 3.7 Quality Checks
- **Sanity Checks**: Loss decreasing, no NaNs, validation tracking.
- **Red Flags**: Overfitting, class imbalance, slow throughput.

## Deliverables
- Trained model checkpoint
- Training curves
- Performance report
- Confusion matrix
- Evaluation notebook
- Log files
