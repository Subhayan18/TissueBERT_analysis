#!/bin/bash
#SBATCH --job-name=cosmos_gpu_test
#SBATCH --account=SNIC2024-22-358  # Replace with your actual project account
#SBATCH --partition=gpua100        # A100 GPU partition (adjust based on actual partition name)
#SBATCH --nodes=1
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:10:00            # 10 minutes test
#SBATCH --output=cosmos_test_%j.log
#SBATCH --error=cosmos_test_%j.err

####################################
# COSMOS GPU Access Test Script
# Purpose: Verify partition access and GPU availability
####################################

echo "=================================================="
echo "COSMOS GPU Access Test - Started"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Date: $(date)"
echo ""

# System Information
echo "=================================================="
echo "1. SYSTEM INFORMATION"
echo "=================================================="
echo "Hostname: $(hostname)"
echo "Operating System: $(cat /etc/os-release | grep PRETTY_NAME)"
echo "Kernel: $(uname -r)"
echo ""

# SLURM Job Details
echo "=================================================="
echo "2. SLURM JOB DETAILS"
echo "=================================================="
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Node List: $SLURM_JOB_NODELIST"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Memory (MB): $SLURM_MEM_PER_NODE"
echo "Time Limit: $SLURM_TIMELIMIT"
echo ""

# CPU Information
echo "=================================================="
echo "3. CPU INFORMATION"
echo "=================================================="
echo "CPU Model:"
lscpu | grep "Model name"
echo ""
echo "CPU Cores Available: $(nproc)"
echo "CPU Architecture:"
lscpu | grep "Architecture\|CPU(s):\|Thread(s) per core\|Core(s) per socket\|Socket(s):"
echo ""

# Memory Information
echo "=================================================="
echo "4. MEMORY INFORMATION"
echo "=================================================="
echo "Total Memory:"
free -h | grep -E "Mem|Swap"
echo ""
echo "Detailed Memory:"
cat /proc/meminfo | grep -E "MemTotal|MemAvailable|MemFree"
echo ""

# GPU Information
echo "=================================================="
echo "5. GPU INFORMATION"
echo "=================================================="

# Check if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
    echo ""
    
    echo "GPU Details:"
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
    echo ""
    
    echo "Full GPU Status:"
    nvidia-smi
    echo ""
    
    echo "CUDA Version:"
    nvcc --version 2>/dev/null || echo "CUDA compiler not in PATH"
    echo ""
else
    echo "ERROR: nvidia-smi not found! GPU not accessible."
    echo "This might indicate the GPU is not allocated properly."
fi
echo ""

# Environment Variables
echo "=================================================="
echo "6. GPU ENVIRONMENT VARIABLES"
echo "=================================================="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-Not set}"
echo "CUDA_HOME: ${CUDA_HOME:-Not set}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-Not set}"
echo ""

# Storage Information
echo "=================================================="
echo "7. STORAGE INFORMATION"
echo "=================================================="
echo "Home Directory Space:"
df -h $HOME
echo ""
echo "Temporary Storage (TMPDIR):"
echo "TMPDIR: ${TMPDIR:-Not set}"
echo "SNIC_TMP: ${SNIC_TMP:-Not set}"
if [ ! -z "$TMPDIR" ]; then
    df -h $TMPDIR
elif [ ! -z "$SNIC_TMP" ]; then
    df -h $SNIC_TMP
fi
echo ""

# Test PyTorch GPU Access (if available)
echo "=================================================="
echo "8. PYTORCH GPU TEST (if available)"
echo "=================================================="

# Try to load Python and test PyTorch
if command -v python3 &> /dev/null; then
    echo "Python version: $(python3 --version)"
    
    # Test if PyTorch is available
    python3 << 'PYEOF'
import sys
print(f"Python executable: {sys.executable}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        
        # Quick GPU test
        print("\nQuick GPU computation test:")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("  Matrix multiplication successful!")
        print(f"  Result shape: {z.shape}")
        print(f"  Device: {z.device}")
    else:
        print("WARNING: PyTorch installed but CUDA not available!")
except ImportError:
    print("PyTorch not installed in current environment")
    print("This is OK for testing partition access")
except Exception as e:
    print(f"Error during PyTorch test: {e}")
PYEOF
else
    echo "Python3 not available"
fi
echo ""

# Performance Test
echo "=================================================="
echo "9. QUICK PERFORMANCE TEST"
echo "=================================================="
echo "Running CPU stress test (5 seconds)..."
timeout 5 stress-ng --cpu 4 --metrics-brief 2>/dev/null || echo "stress-ng not available (OK)"
echo ""

# Module System Check
echo "=================================================="
echo "10. MODULE SYSTEM CHECK"
echo "=================================================="
if command -v module &> /dev/null; then
    echo "Available GPU/ML related modules:"
    module avail 2>&1 | grep -iE "(cuda|pytorch|tensorflow|gpu)" | head -20
else
    echo "Module system not available or not initialized"
fi
echo ""

# Network Test
echo "=================================================="
echo "11. NETWORK CONNECTIVITY"
echo "=================================================="
echo "Checking InfiniBand (if available):"
ibstat 2>/dev/null | grep -E "State|Rate" || echo "InfiniBand not available or ibstat not installed"
echo ""

# Summary
echo "=================================================="
echo "TEST SUMMARY"
echo "=================================================="
echo "✓ Job submitted and running on: $(hostname)"
echo "✓ Partition: $SLURM_JOB_PARTITION"
echo "✓ CPUs allocated: $(nproc)"
echo "✓ Memory available: $(free -h | grep Mem | awk '{print $2}')"

if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "✓ GPUs detected: $GPU_COUNT x $GPU_NAME"
else
    echo "✗ GPUs NOT detected"
fi

echo ""
echo "=================================================="
echo "Test Completed Successfully!"
echo "End Time: $(date)"
echo "=================================================="

# Create a simple status file
echo "Test completed on $(date)" > cosmos_test_${SLURM_JOB_ID}_status.txt
echo "Partition: $SLURM_JOB_PARTITION" >> cosmos_test_${SLURM_JOB_ID}_status.txt
echo "Node: $(hostname)" >> cosmos_test_${SLURM_JOB_ID}_status.txt
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" >> cosmos_test_${SLURM_JOB_ID}_status.txt
fi

exit 0
