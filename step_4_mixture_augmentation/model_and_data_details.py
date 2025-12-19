import torch
import sys
sys.path.insert(0, '/home/chattopa/data_storage/TissueBERT_analysis/step_4_mixture_augmentation')

checkpoint = torch.load('/home/chattopa/data_storage/MethAtlas_WGBSanalysis/mixture_deconvolution_results/phase2_multitissue/checkpoints/checkpoint_best.pt', map_location='cpu')

print("Checkpoint keys:", list(checkpoint.keys()))
print("\nConfig:", checkpoint.get('config', 'NO CONFIG FOUND'))
print("\nModel architecture info (if available):")
if 'config' in checkpoint:
    for key, val in checkpoint['config'].items():
        print(f"  {key}: {val}")
