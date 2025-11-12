#!/bin/bash
# Load modules for Step 1.2

module reset
module load GCC/12.3.0
module load SciPy-bundle/2023.07 
module load matplotlib/3.7.2
module load OpenMPI/4.1.5
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load JupyterNotebook/7.0.2
module load scikit-build/0.17.6
module load imageio/2.33.1
module load Python/3.11.3

module list
