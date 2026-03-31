#!/bin/bash
#SBATCH --job-name=param_recovery
#SBATCH --output=/users/azhan378/scratch/logs/slurm-%j-%a.log
#SBATCH --error=/users/azhan378/scratch/logs/slurm-%j-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --array=0-150
#SBATCH --partition=gpu
#SBATCH --account=carney-frankmj-condo

# Parameter recovery inference sweep
# This script runs parameter recovery inference using pre-generated data (sim_ids 0-250)
# Run data_generation.sbatch FIRST to generate the synthetic data
# Usage: sbatch param_recovery.sbatch

# Configuration
OUTPUT_DIR="/users/azhan378/scratch/param_recovery_x0_simulate_150"
BASE_SEED=42
SCRIPT_PATH="${SLURM_SUBMIT_DIR}/param_recovery_main.py"

# Create logs directory
mkdir -p "/users/azhan378/scratch/logs/param_sets_x0_simulate_150"
mkdir -p "${OUTPUT_DIR}"

# Get simulation ID from array index
SIM_ID=$SLURM_ARRAY_TASK_ID
SEED=$((BASE_SEED + SIM_ID))

# Print job information
echo "==============================================="
echo "Parameter Recovery Simulation"
echo "==============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Simulation ID: $SIM_ID"
echo "Random Seed: $SEED"
echo "Output Directory: $OUTPUT_DIR"
echo "CUDA Visible Devices: $CUDA_VISIBLE_DEVICES"
echo "==============================================="


# Set JAX to use both GPUs
export JAX_PLATFORM_NAME=gpu
export XLA_FLAGS="--xla_force_host_platform_device_count=2"

# Parameter sampling configuration
SIMULATE_ETA=FALSE
ETA_CONST=0.3

SIMULATE_KAPPA=FALSE
KAPPA_CONST=0.5

SIMULATE_A=FALSE
A_CONST=2.0

SIMULATE_X0=TRUE
X0_CONST=0.5

# Run the parameter recovery simulation
python "$SCRIPT_PATH" \
    --sim-id "$SIM_ID" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --simulate-eta "$SIMULATE_ETA" \
    --eta-const "$ETA_CONST" \
    --simulate-kappa "$SIMULATE_KAPPA" \
    --kappa-const "$KAPPA_CONST" \
    --simulate-a "$SIMULATE_A" \
    --a-const "$A_CONST" \
    --simulate-x0 "$SIMULATE_X0" \
    --x0-const "$X0_CONST"

# Check exit code
if [ $? -eq 0 ]; then
    echo "Simulation $SIM_ID completed successfully"
else
    echo "Simulation $SIM_ID failed with exit code $?"
    exit 1
fi
