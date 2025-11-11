# Phase 4: Validation & Testing

## Goal
Evaluate model performance on held-out data and simulated cfDNA mixtures.

## Substeps

### 4.1 Test Set Evaluation
- **Data**: 13 held-out pure tissue samples.
- **Metrics**: Accuracy >85%, F1 >0.80, top-3 accuracy >95%, confidence calibration.

### 4.2 Simulated Mixture Testing
- **Method**: Mix 2–5 tissue types with known proportions.
- **Evaluation**: MAE per tissue, correlation with true proportions, detection of trace components (<5%).

### 4.3 GTEx Validation (Optional)
- **Goal**: Test generalization to array-based data.
- **Challenge**: Distribution shift from WGBS to EPIC arrays.

### 4.4 Baseline Comparisons
- **Methods**: Houseman, NNLS, feedforward NN.
- **Goal**: Demonstrate DNABERT-S superiority.

## Deliverables
- Test set performance report
- Mixture validation results
- Baseline comparison table
- Error analysis
- Confidence analysis
