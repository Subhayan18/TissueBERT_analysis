# Quick Decision Reference

**One-page summary of key design decisions and their justifications.**

---

## Architecture: DNABERT-S (6 layers, 512 hidden)
**Why?** Balance of capacity and data scale. Transformers capture long-range CpG co-methylation. 20M parameters prevents overfitting on 765 samples.

**Not CNNs:** Limited receptive field for 150bp sequences.  
**Not larger models:** Risk overfitting, no clear benefit with limited data.

---

## Loss: Cross-Entropy + Label Smoothing (0.1)
**Why?** Standard, stable, well-understood. Label smoothing prevents overconfidence and improves probability calibration for deconvolution.

**Not weighted CE:** Data-level balancing (sampling) more effective.  
**Not focal loss:** Added complexity, hyperparameter sensitive.

---

## Optimizer: AdamW (4e-4, weight decay 0.01)
**Why?** Adaptive learning works better for transformers than SGD. Decoupled weight decay for proper L2 regularization.

**Learning rate:** From LR range test, higher than fine-tuning (training from scratch), lower than pre-training (limited data).

---

## Scheduler: OneCycleLR (10% warmup, cosine annealing)
**Why?** Warmup prevents early instability. High LR escapes sharp minima. Cosine annealing settles into wide minima for generalization.

**Not constant:** Suboptimal convergence.  
**Not step decay:** Abrupt changes destabilize training.

---

## Data: Tissue-Balanced Sampling + CpG Dropout (5%)
**Why balanced?** Blood is 27% of data. File-level weights ensure each tissue appears equally in batches. Avoids PyTorch 2^24 limit.

**Why CpG dropout?** Simulates variable cfDNA coverage, prevents overfitting, improves robustness to missing data.

---

## Batch: Size 128, Gradient Accumulation 4x (effective=512)
**Why?** Large effective batch for stable gradients. Gradient accumulation fits in GPU memory. More frequent updates than batch=512.

**Not batch=512:** GPU OOM (~40GB needed).  
**Not batch=32:** Unstable gradients, poor generalization.

---

## Resources: 32 CPUs, 128GB RAM, 24 Workers
**Why?** Bottleneck is data loading, not computation. 24 workers eliminate GPU idle time (95% utilization vs 80%). 20% faster training.

**Memory:** 24 workers × 4GB + 32GB buffer = 128GB.  
**Rule:** workers = CPUs × 0.75

---

## Checkpointing: Every 5 Epochs + Best Val Loss/Acc + Auto-Resume (max 5x)
**Why multiple?** Best loss ≠ best accuracy. Want both for different uses.

**Why every 5?** Balance storage (25GB for 50 epochs) vs resume granularity.

**Auto-resume:** 120h job limit, ~25h for 50 epochs. May timeout. Auto-resubmits, no manual intervention. Limit 5x prevents infinite loops.

---

## Logging: TensorBoard + CSV
**Why both?** TensorBoard for real-time visualization. CSV for programmatic analysis.

**Per-tissue metrics:** 22 classes—overall accuracy hides poor tissues. Critical for debugging imbalance.

---

## Training: 20 → 50 → 100 Epochs (Staged)
**Why staged?** Proof-of-concept (20 epochs) validates setup. Full training (50 epochs) if successful. Extended (100 epochs) only if continued improvement.

**Resource efficiency:** Don't waste 80h if setup is wrong.

---

## Validation: Every 1 Epoch
**Why?** ~30 min per epoch, ~3 min validation. 10% overhead. Early overfitting detection.

**Not every N steps:** Too expensive (135 files × 51k regions).

---

## Classification Not Deconvolution
**Why?** Simpler (one label per sample). Leverages pure tissue data. Standard approach (MethylBERT, CancerDetector).

**Deconvolution:** Aggregate region predictions → tissue proportions. Defer to inference.

---

## Key Design Principles

1. **Simplicity first:** Standard methods before exotic techniques
2. **Fail fast:** 20 epoch validation before committing to 50+
3. **Production-ready:** Auto-resumption, error handling, comprehensive logging
4. **Biological grounding:** CpG dropout, tissue balancing match real biology
5. **Interpretability:** Per-tissue metrics, confusion matrices

---

## What We DIDN'T Do (And Why)

- ❌ **Pre-trained models:** Unmethylated pre-training doesn't transfer
- ❌ **Weighted loss:** Data-level balancing more effective
- ❌ **Focal loss:** Unnecessary complexity for first iteration
- ❌ **Sequence augmentation:** Destroys biological signal
- ❌ **Mixed precision:** FP32 sufficient, FP16 adds instability risk
- ❌ **Multi-GPU:** Single GPU sufficient, save for later if needed

---

**Use this for:** Quick refresher on design decisions when modifying code or tuning hyperparameters.

**Full justifications:** See README.md
