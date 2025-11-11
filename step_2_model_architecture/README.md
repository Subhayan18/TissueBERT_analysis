# Phase 2: Model Architecture

## Goal
Design and implement the DNABERT-S model for tissue deconvolution from cfDNA methylation patterns.

## Substeps

### 2.1 Architecture Design
- **Model**: DNABERT-S (6-layer transformer, 512 hidden dim).
- **Rationale**: Efficient for short reads, faster training, suitable for panel-specific data.

### 2.2 Model Components
- **Inputs**: DNA tokens (3-mers), methylation states, positional embeddings.
- **Encoder**: Transformer with multi-head attention and feed-forward layers.
- **Output**: 39-dimensional softmax for tissue proportions.

### 2.3 Input Representation
- **DNA Tokens**: 64 3-mers + 5 special tokens.
- **Methylation States**: Unmethylated, methylated, non-CpG.
- **Padding**: Right-pad to 150 tokens.

### 2.4 Loss Function
- **Primary**: Cross-entropy for pure tissue samples.
- **Alternative**: KL divergence for cfDNA mixtures.

### 2.5 Pre-training (Optional)
- **Objective**: Masked language modeling on DNA sequences.
- **Benefit**: Improves motif recognition and fine-tuning performance.

## Deliverables
- Model implementation (PyTorch)
- Input preprocessing pipeline
- Loss function modules
- Configuration files
- Architecture diagram
- Unit tests
