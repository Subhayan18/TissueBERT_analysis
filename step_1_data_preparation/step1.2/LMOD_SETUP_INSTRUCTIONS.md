# LMOD Module Setup for Step 1.2

This guide will help you load the required LMOD modules for Step 1.2 (Simulate Read-Level Training Data).

## Overview

Instead of using conda, we'll use LMOD to load pre-installed Python and scientific computing libraries. This is often more efficient on HPC systems.

## Step 1: Reset modules (clean slate)

Start with a clean module environment:

```bash
module reset
```

This ensures no conflicting modules are loaded.

## Step 2: Load required modules

Load the modules in this specific order:

```bash
module load GCC/12.3.0
module load SciPy-bundle/2023.07 
module load matplotlib/3.7.2
module load OpenMPI/4.1.5
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load JupyterNotebook/7.0.2
module load scikit-build/0.17.6
module load imageio/2.33.1
module load Python/3.11.3
```

### What each module provides

| Module | Provides | Required for Step 1.2? |
|--------|----------|------------------------|
| `GCC/12.3.0` | GNU Compiler Collection | ✅ Yes (base dependency) |
| `SciPy-bundle/2023.07` | numpy, pandas, scipy | ✅ Yes (core packages) |
| `matplotlib/3.7.2` | Plotting library | ⚠️ Optional (for future visualization) |
| `OpenMPI/4.1.5` | MPI for parallel computing | ⚠️ Optional (not used in Step 1.2) |
| `PyTorch/2.1.2` | Deep learning framework | ⚠️ Optional (for future steps) |
| `JupyterNotebook/7.0.2` | Interactive notebooks | ⚠️ Optional (for analysis) |
| `scikit-build/0.17.6` | Build system | ⚠️ Optional (for package installation) |
| `imageio/2.33.1` | Image I/O | ⚠️ Optional (not used in Step 1.2) |
| `Python/3.11.3` | Python interpreter | ✅ Yes (main interpreter) |

### Minimal modules for Step 1.2 only

If you only want to run Step 1.2 scripts, you can load just these:

```bash
module reset
module load GCC/12.3.0
module load SciPy-bundle/2023.07
module load Python/3.11.3
```

This provides everything needed for scripts 01, 02, and 03.

## Step 3: Verify module loading

Check that modules are loaded correctly:

```bash
module list
```

You should see all loaded modules listed.

## Step 4: Verify Python and packages

Test that Python and required packages are available:

```bash
python --version
# Should show: Python 3.11.3

python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import pandas; print('pandas:', pandas.__version__)"
python -c "import scipy; print('scipy:', scipy.__version__)"
```

Expected output:
```
Python 3.11.3
numpy: 1.25.1
pandas: 2.0.3
scipy: 1.11.1
```

## Step 5: Test environment (optional but recommended)

Run a comprehensive test:

```bash
python << 'EOF'
import numpy as np
import pandas as pd
from scipy import sparse
import sys

print("Testing environment...")
print(f"✓ Python: {sys.version}")
print(f"✓ numpy loaded: {np.__version__}")
print(f"✓ pandas loaded: {pd.__version__}")
print(f"✓ scipy loaded")

# Quick functionality test
test_array = np.random.rand(10, 10)
test_df = pd.DataFrame(test_array)
print("✓ Basic operations work")
print("\nEnvironment setup complete!")
EOF
```

## Step 6: Create a module loading script (recommended)

Create a file `load_modules.sh` for easy reloading:

```bash
cat > load_modules.sh << 'EOF'
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

echo "Modules loaded successfully!"
module list
EOF

chmod +x load_modules.sh
```

Usage:
```bash
source load_modules.sh
```

## Troubleshooting

### If module load fails

Check if module exists:

```bash
module avail GCC
module avail Python
module avail SciPy
```

This will show all available versions.

### If you get "module: command not found"

LMOD should be available by default on HPC systems. If not, contact your system administrator.

### If Python packages are not found after loading modules

Make sure you loaded `SciPy-bundle/2023.07`:

```bash
module list | grep SciPy
```

If not loaded:
```bash
module load SciPy-bundle/2023.07
```

### Module conflicts

If you get module conflict errors:

```bash
module reset
# Then reload all modules
```

### Check what's in SciPy-bundle

```bash
module show SciPy-bundle/2023.07
```

This will display all packages included in the bundle.

## Using Modules in Interactive Sessions

### For interactive work

```bash
# Start interactive session
salloc --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=2:00:00

# Load modules
source load_modules.sh

# Run Python
python
```

### For Jupyter Notebook

```bash
# Load modules
source load_modules.sh

# Start Jupyter
jupyter notebook --no-browser --port=8888
```

## For SLURM Jobs

The module loading is already included in `run_step1.2.sh`. The script automatically:
1. Runs `module reset`
2. Loads all required modules
3. Verifies Python and packages
4. Executes the Python scripts

You don't need to manually load modules when submitting the SLURM job!

## Advantages of LMOD vs Conda

### Why LMOD for HPC?

✅ **Pre-optimized**: Libraries compiled for your specific HPC architecture  
✅ **No installation**: Packages already installed by system administrators  
✅ **Faster loading**: No environment activation overhead  
✅ **Better performance**: Optimized with system-specific compiler flags  
✅ **Version consistency**: Centrally managed versions  
✅ **Less disk usage**: Shared installations across users

### When to use conda instead?

- Need packages not available in LMOD
- Need specific package versions not in LMOD
- Working on local workstation (not HPC)
- Need isolated environments for different projects

## Package Versions

The `SciPy-bundle/2023.07` module includes:

| Package | Version |
|---------|---------|
| numpy | 1.25.1 |
| scipy | 1.11.1 |
| pandas | 2.0.3 |
| matplotlib | 3.7.2 |
| scikit-learn | 1.3.0 |
| And many more... | |

To see complete list:
```bash
pip list
```

## Quick Start Command Sequence

Copy and paste these commands to verify everything works:

```bash
module reset
module load GCC/12.3.0
module load SciPy-bundle/2023.07
module load Python/3.11.3
python -c "import numpy, pandas, scipy; print('All packages available!')"
```

## Persistent Module Loading

### Option 1: Add to your .bashrc (not recommended)

This will load modules every time you log in:

```bash
echo "source $HOME/scripts/step1.2/load_modules.sh" >> ~/.bashrc
```

**Warning**: This can slow down login and cause conflicts.

### Option 2: Load manually when needed (recommended)

```bash
source load_modules.sh
```

### Option 3: Create an alias

Add to your `.bashrc`:
```bash
alias loadstep12="source $HOME/scripts/step1.2/load_modules.sh"
```

Then use:
```bash
loadstep12
```

## Next Steps

After verifying modules load correctly:
1. Return to the main `README.md`
2. Follow the installation instructions
3. Submit your SLURM job with `sbatch run_step1.2.sh`

## Module Information Commands

Useful commands for working with LMOD:

```bash
# List currently loaded modules
module list

# Search for available modules
module avail python
module avail scipy

# Show module details
module show Python/3.11.3

# Unload a specific module
module unload Python

# Unload all modules
module purge

# Reset to default modules
module reset

# Save current module collection
module save mycollection

# Restore saved collection
module restore mycollection
```

## For System Administrators

If you're setting up LMOD for others, ensure these modules are available:
- GCC/12.3.0 or later
- Python/3.11.3 or later
- SciPy-bundle/2023.07 (includes numpy, pandas, scipy)

Build with optimizations for your CPU architecture for best performance.
