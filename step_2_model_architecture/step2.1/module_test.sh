#!/bin/bash
#SBATCH --job-name=module_test
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=module_test_%j.log

echo "Loading modules..."
module load GCC/12.3.0  
module load OpenMPI/4.1.5 
module load CUDA-Python/12.1.0-CUDA-12.1.1 
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load Transformers/4.39.3
module load tensorboard/2.15.1
module load SciPy-bundle/2023.07 
module load matplotlib/3.7.2
module load scikit-learn/1.3.1
module load h5py/3.9.0
module load Python/3.11.3

# export PYTHONPATH="/home/chattopa/data_storage/user_libs:$PYTHONPATH"

echo "================================================"
echo "PYTHON PACKAGE VERIFICATION"
echo "================================================"

python3 << 'EOF'
import sys
print(f"Python: {sys.version}")
print("\nChecking packages...\n")

packages = {
    'Core ML': ['torch', 'transformers', 'sklearn'],
    'Scientific': ['numpy', 'pandas', 'scipy', 'matplotlib'],
    'Data Handling': ['h5py'],
    'Optional': ['wandb', 'accelerate', 'datasets', 'einops', 'pyarrow']
}

for category, pkgs in packages.items():
    print(f"{category}:")
    for pkg in pkgs:
        try:
            module = __import__(pkg)
            version = getattr(module, '__version__', 'unknown')
            print(f"  ✓ {pkg:15s} {version}")
        except ImportError:
            print(f"  ✗ {pkg:15s} NOT FOUND")
    print()

# Test PyTorch GPU
import torch
print(f"PyTorch GPU Test:")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU count: {torch.cuda.device_count()}")
    print(f"  GPU name: {torch.cuda.get_device_name(0)}")
    
    # Quick computation test
    x = torch.randn(100, 100).cuda()
    y = torch.matmul(x, x)
    print(f"  ✓ GPU computation successful")
else:
    print(f"  ✗ CUDA not available!")

# Test Transformers
from transformers import AutoTokenizer, AutoModel
print(f"\nTransformers Test:")
print(f"  ✓ Can import AutoTokenizer and AutoModel")

EOF

echo "================================================"
echo "Check completed!"
echo "================================================"
