#!/bin/bash
# Quick Evaluation Script for Phase 1
# Run this after Phase 1 training completes

cd /home/chattopa/data_storage/TissueBERT_analysis/step_4_mixture_augmentation

# Load environment
source /home/chattopa/data_storage/TissueBERT_analysis/step_2_model_architecture/LMOD.sourceme

# Run evaluation
python3 evaluate_deconvolution.py \
    --checkpoint /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase1_2tissue/checkpoints/checkpoint_best.pt \
    --test_h5 /home/chattopa/data_storage/MethAtlas_WGBSanalysis/training_dataset/mixture_data/phase1_test_mixtures.h5 \
    --output_dir /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase1_2tissue/evaluation \
    --device cuda \
    --batch_size 32

echo ""
echo "Evaluation complete!"
echo "Results saved to: /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase1_2tissue/evaluation"
echo ""
echo "View report:"
echo "  cat /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase1_2tissue/evaluation/evaluation_report.txt"
echo ""
echo "View figures:"
echo "  ls /home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase1_2tissue/evaluation/figures/"
