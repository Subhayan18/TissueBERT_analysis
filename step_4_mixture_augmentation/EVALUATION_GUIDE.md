# Evaluation Script Usage Guide

## Overview

`evaluate_deconvolution.py` generates comprehensive evaluation reports and detailed visualizations for trained mixture deconvolution models.

**Works for all phases (1, 2, 3).**

---

## Usage

### Basic Command

```bash
python3 evaluate_deconvolution.py \
    --checkpoint /path/to/checkpoint_best.pt \
    --test_h5 /path/to/test_mixtures.h5 \
    --output_dir /path/to/evaluation_results \
    --device cuda \
    --batch_size 32
```

### Phase 1 Example

```bash
python3 evaluate_deconvolution.py \
    --checkpoint /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase1_2tissue/checkpoints/checkpoint_best.pt \
    --test_h5 /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/mixture_data/phase1_test_mixtures.h5 \
    --output_dir /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase1_2tissue/evaluation
```

### Phase 2 Example

```bash
python3 evaluate_deconvolution.py \
    --checkpoint /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase2_multitissue/checkpoints/checkpoint_best.pt \
    --test_h5 /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/mixture_data/phase2_test_mixtures.h5 \
    --output_dir /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase2_multitissue/evaluation
```

### Phase 3 Example

```bash
python3 evaluate_deconvolution.py \
    --checkpoint /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase3_realistic/checkpoints/checkpoint_best.pt \
    --test_h5 /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/mixture_data/phase3_test_mixtures.h5 \
    --output_dir /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase3_realistic/evaluation
```

---

## Generated Outputs

### Text Reports

1. **`evaluation_report.txt`** - Comprehensive text report with:
   - Overall performance metrics
   - Per-tissue performance table
   - Mixture type analysis
   - Best/worst predictions
   - Success criteria evaluation

2. **`metrics.json`** - All metrics in JSON format

3. **`per_tissue_metrics.csv`** - Detailed per-tissue statistics

4. **`mixture_type_metrics.csv`** - Performance by mixture complexity

5. **`predictions.npy`, `true_proportions.npy`** - Raw arrays for further analysis

---

### Figures (10 detailed plots)

#### `fig1_overall_performance.png`
- **Scatter plot**: Predicted vs True proportions (all data points)
- **Hexbin plot**: Density visualization
- Shows MAE, R² on plots

#### `fig2_error_analysis.png`
- **Error histogram**: Distribution of (Pred - True)
- **Absolute error histogram**: Distribution of |Pred - True|
- **Per-sample MAE**: Distribution across samples
- **Residual plot**: Residuals vs predicted values

#### `fig3_per_tissue_scatter.png`
- **Grid of scatter plots**: One per tissue (22 total)
- Each shows: Predicted vs True for that tissue
- Annotated with MAE, R², sample count
- Only shows samples where tissue is present

#### `fig4_per_tissue_mae.png`
- **Horizontal bar chart**: MAE per tissue (sorted)
- Color-coded: Green (<5%), Orange (5-10%), Red (>10%)
- Annotated with sample counts
- Shows 5% and 10% thresholds

#### `fig5_tissue_frequency.png`
- **Bar chart**: How often each tissue appears
- **Scatter plot**: MAE vs Frequency
  - Shows if rare tissues have higher errors

#### `fig6_heatmaps.png`
- **True proportions heatmap**: First 100 samples
- **Predicted proportions heatmap**: First 100 samples
- **Error heatmap**: (Predicted - True) with diverging colormap
- Tissues on Y-axis, samples on X-axis

#### `fig7_correlation_heatmap.png`
- **True correlations**: Tissue co-occurrence patterns
- **Predicted correlations**: Model's learned tissue relationships
- Shows which tissues appear together frequently

#### `fig8_best_predictions.png`
- **Top 5 best predictions**: Bar charts comparing true vs predicted
- Shows tissue names on X-axis
- Helps understand what model does well

#### `fig9_worst_predictions.png`
- **Top 5 worst predictions**: Bar charts comparing true vs predicted
- Shows where model struggles
- Useful for identifying failure modes

#### `fig10_mixture_complexity.png`
- **MAE vs complexity**: How error changes with more tissues
- **RMSE vs complexity**: Root mean squared error trends
- **R² vs complexity**: Goodness of fit trends
- X-axis: Number of tissues in mixture (2, 3, 4, 5, etc.)

---

## Metrics Explained

### Overall Metrics

**MAE (Mean Absolute Error)**:
- Average error per tissue proportion
- Example: MAE=0.025 means 2.5% average error
- **Lower is better**

**RMSE (Root Mean Squared Error)**:
- Penalizes large errors more than small ones
- Always ≥ MAE
- **Lower is better**

**R² (R-squared)**:
- Goodness of fit (how much variance explained)
- Range: -∞ to 1.0
- 1.0 = perfect fit, 0 = no better than mean
- **Higher is better**

**Pearson/Spearman Correlation**:
- Linear/rank correlation between predicted and true
- Range: -1 to 1
- 1 = perfect positive correlation
- **Higher is better** (closer to 1)

### Per-Tissue Metrics

**MAE (when present)**:
- Error only on samples where tissue is present
- More meaningful than MAE on all samples

**Frequency**:
- Proportion of test samples containing this tissue
- Helps interpret reliability (more samples = more reliable)

**Bias**:
- Average (Predicted - True) for present samples
- Positive = overestimation, Negative = underestimation

---

## Interpreting Results

### Success Criteria

**Phase 1 (2-tissue)**:
- ✓ Target: MAE < 5% (0.05)
- ✓ Expected: ~2-4% based on your training

**Phase 2 (3-5 tissue)**:
- ✓ Target: MAE < 8% (0.08)
- ✓ Expected: ~4-6%

**Phase 3 (Realistic cfDNA)**:
- ✓ Target: MAE < 10% (0.10)
- ✓ Expected: ~6-8%

### What to Look For

**Good Signs**:
- MAE close to or below target
- R² > 0.8
- Predictions cluster along diagonal (fig1)
- Errors centered around zero (fig2)
- Most tissues have MAE < 10% (fig4)
- Proportions sum to 1.0 (validation check)

**Potential Issues**:
- High MAE on specific tissues → May need more training data
- Systematic bias (predictions consistently high/low) → Check loss function
- Poor performance on rare tissues → Expected, check frequency
- Degrading performance with complexity → May need more Phase 2/3 training

---

## Runtime

- **Phase 1** (500 test samples): ~1-2 minutes
- **Phase 2** (1,000 test samples): ~2-3 minutes
- **Phase 3** (1,500 test samples): ~3-4 minutes

Most time spent on inference and generating plots.

---

## Tips

1. **Compare across phases**: Run evaluation on all 3 phases and compare results
2. **Check tissue-specific errors**: Some tissues harder to predict (e.g., rare ones)
3. **Look at worst predictions**: Understand failure modes
4. **Validate proportions sum**: Should always be ≈1.0
5. **Use correlation heatmap**: Understand tissue relationships learned by model

---

## Example Workflow

```bash
# After Phase 1 training completes
cd /home/chattopa/data_storage/TissueBERT_analysis/step_4_mixture_augmentation

# Run evaluation
python3 evaluate_deconvolution.py \
    --checkpoint /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase1_2tissue/checkpoints/checkpoint_best.pt \
    --test_h5 /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/mixture_data/phase1_test_mixtures.h5 \
    --output_dir /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase1_2tissue/evaluation

# Check results
cat /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase1_2tissue/evaluation/evaluation_report.txt

# View figures
ls /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase1_2tissue/evaluation/figures/
```

---

## Troubleshooting

**"ModuleNotFoundError: model_deconvolution"**
- Make sure you're in the directory with `model_deconvolution.py`
- Or add to Python path: `export PYTHONPATH=/path/to/scripts:$PYTHONPATH`

**"CUDA out of memory"**
- Reduce batch size: `--batch_size 16` or `--batch_size 8`
- Or use CPU: `--device cpu` (slower but works)

**"FileNotFoundError" for checkpoint**
- Verify checkpoint path exists
- Check that training completed successfully

**Missing figures**
- Check `figures/` subdirectory in output_dir
- Ensure matplotlib/seaborn installed

---

## Quick Reference

```bash
# Minimal command
python3 evaluate_deconvolution.py \
    --checkpoint checkpoint_best.pt \
    --test_h5 test_mixtures.h5 \
    --output_dir evaluation_results

# With all options
python3 evaluate_deconvolution.py \
    --checkpoint checkpoint_best.pt \
    --test_h5 test_mixtures.h5 \
    --output_dir evaluation_results \
    --device cuda \
    --batch_size 32
```

---

**The evaluation script provides everything you need to thoroughly assess model performance and generate publication-quality figures!**
