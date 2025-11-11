# Phase 5: Clinical Application

## Goal
Apply the trained model to real PDAC cfDNA samples and assess clinical utility.

## Substeps

### 5.1 PDAC Sample Processing
- **Cohort**: 5 patients, ~3 timepoints each.
- **Timepoints**: Diagnosis, post-surgery, MRD monitoring, pre-metastasis.

### 5.2 Inference Pipeline
- **Flow**: FASTQ → alignment → methylation calling → model input → inference → report.

### 5.3 Clinical Interpretation
- **Analysis**:
  - Quantify tissue proportions
  - Track changes over time
  - Identify elevated signals
  - Predict metastasis

### 5.4 Expected Outcomes
- **Success Criteria**:
  - Stable baselines
  - Temporal tracking
  - Predictive signals in ≥2 patients
  - Lead time: 2–6 months before imaging

### 5.5 Validation Expansion
- **Next Steps**:
  - Expand to 30 patients
  - Multi-center validation
  - Prospective study
  - Publication preparation

## Deliverables
- Inference pipeline
- Patient-level reports
- Clinical correlation analysis
- Case studies
- Manuscript draft
