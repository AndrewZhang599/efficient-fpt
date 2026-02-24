#!/bin/bash
#SBATCH --job-name=data_gen
#SBATCH --output=/users/azhan378/scratch/logs/data_gen-%j-%a.log
#SBATCH --error=/users/azhan378/scratch/logs/data_gen-%j-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=10:00:00
#SBATCH --array=0-250
#SBATCH --partition=batch
#SBATCH --account=carney-frankmj-condo

# Data generation parameter sweep
# This script generates synthetic data for sim_ids 0-250 (parallelizable)
# Usage: sbatch data_generation.sbatch

# Configuration
OUTPUT_DIR="/users/azhan378/scratch/param_recovery_sample_kappa"
BASE_SEED=42
SCRIPT_PATH="${SLURM_SUBMIT_DIR}/data_generation.py"

# Create logs directory
mkdir -p "/users/azhan378/scratch/logs"
mkdir -p "${OUTPUT_DIR}"

# Get simulation ID from array index
SIM_ID=$SLURM_ARRAY_TASK_ID
SEED=$((BASE_SEED + SIM_ID))

# Print job information
echo "==============================================="
echo "Data Generation"
echo "==============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Simulation ID: $SIM_ID"
echo "Random Seed: $SEED"
echo "Output Directory: $OUTPUT_DIR"
echo "==============================================="


# Run the data generation
python "$SCRIPT_PATH" \
    --sim-id "$SIM_ID" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED"

# Check exit code
if [ $? -eq 0 ]; then
    echo "Data generation for simulation $SIM_ID completed successfully"
else
    echo "Data generation for simulation $SIM_ID failed with exit code $?"
    exit 1
fi
