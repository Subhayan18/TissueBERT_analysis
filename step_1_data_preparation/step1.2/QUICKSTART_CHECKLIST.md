# Step 1.2 Quick-Start Checklist

Follow these steps in order:

## Setup Steps

- [ ] **STEP 1**: Download all files
  - `01_inspect_data.py`
  - `02_simulate_reads.py`
  - `03_verify_output.py`
  - `run_step1.2.sh`
  - `LMOD_SETUP_INSTRUCTIONS.md`
  - `README.md`

- [ ] **STEP 2**: Upload to your secure server
  ```bash
  mkdir -p $HOME/scripts/step1.2
  cd $HOME/scripts/step1.2
  # Upload all files here
  ```

- [ ] **STEP 3**: Make SLURM script executable
  ```bash
  chmod +x run_step1.2.sh
  ```

- [ ] **STEP 4**: Create logs directory
  ```bash
  mkdir -p logs
  ```

- [ ] **STEP 5**: Verify LMOD modules are available
  ```bash
  module avail Python
  module avail SciPy-bundle
  # Should show Python/3.11.3 and SciPy-bundle/2023.07
  ```

- [ ] **STEP 6**: Test module loading
  ```bash
  module reset
  module load GCC/12.3.0
  module load SciPy-bundle/2023.07
  module load Python/3.11.3
  python -c "import numpy, pandas, scipy; print('Success!')"
  ```

- [ ] **STEP 7**: Submit SLURM job
  ```bash
  sbatch run_step1.2.sh
  ```

- [ ] **STEP 8**: Monitor job
  ```bash
  # Get job ID from submission
  squeue -u $USER
  tail -f logs/step1.2_<jobid>.out
  ```

- [ ] **STEP 9**: When complete, check outputs
  ```bash
  ls -lh /home/chattopa/data_storage/MethAtlas_WGBSanalysis/synthetic_reads/
  ```
  
  Should see:
  - 82 `sample_*.npz` files (~100-300 MB each)
  - `tissue_labels.txt`
  - `simulation_summary.txt`
  - `sample_statistics.csv`

- [ ] **STEP 10**: Review quality report
  ```bash
  cat /home/chattopa/data_storage/MethAtlas_WGBSanalysis/synthetic_reads/simulation_summary.txt
  ```
  
  Look for: `✓ All checks passed successfully!`

- [ ] **STEP 11**: Ready for Step 1.3!

## Expected Timeline

| Script | Duration |
|--------|----------|
| Script 01 | ~1 minute |
| Script 02 | ~10-20 hours ⏰ **THIS IS THE LONG ONE** |
| Script 03 | ~5 minutes |
| **Total** | **~10-20 hours** |

## Quick Troubleshooting

### Problem: Job not starting
- **Check**: `squeue -u $USER`
- **Solution**: Contact sysadmin if queue issues

### Problem: Job failed
- **Check**: `logs/step1.2_<jobid>.err`
- **Common causes**:
  - Missing input file
  - LMOD modules not available
  - Out of memory (increase `--mem`)

### Problem: Scripts not in right place
- SLURM script expects: `$HOME/scripts/step1.2/`
- Edit line 83 of `run_step1.2.sh` if different location

### Problem: "ModuleNotFoundError" for packages
- **Check**: `module list | grep SciPy`
- **Solution**: Verify SciPy-bundle/2023.07 is loaded

## Important Notes

✅ This generates **~2 BILLION synthetic reads**  
✅ Will take **10-20 hours** - this is NORMAL  
✅ Uses **~250 GB RAM** during processing  
✅ Output will be **~10-20 GB compressed**  
✅ Make sure you have enough disk space!

## Getting Help

1. Read `README.md` (comprehensive guide)
2. Check `LMOD_SETUP_INSTRUCTIONS.md` (module setup)
3. Review log files in `logs/` directory
4. Check error file: `logs/step1.2_<jobid>.err`
5. Verify modules: `module list`
