#!/bin/bash
#SBATCH --job-name=pmc_compare
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Notes:
# - For JAX GPU runs, override partition/gres on submission, e.g.:
#     sbatch --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G run_pymc_sampler_comparison.sbatch jax --draws 500 --tune 200
# - For CPU Cython runs:
#     sbatch --partition=batch --cpus-per-task=8 --mem=32G run_pymc_sampler_comparison.sbatch cython --draws 500 --tune 200
#
# The first positional argument is MODE: cython | jax | both
# Additional arguments are passed through to the python script.

set -euo pipefail

MODE="${1:-cython}"
shift || true

mkdir -p logs

# Optional environment knobs (uncomment if you see JAX memory/preallocation issues)
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# If your repo uses a local src/ layout, set this (or pass --src-path ...)
# export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

# Load modules / activate env here (cluster-specific). Examples:
# module purge
# module load python/3.11
# module load cuda/12.4
# source /path/to/venv/bin/activate

DATA_PATH="${DATA_PATH:-example4_new/addm_data_20251015-163921.pkl}"

python pymc_sampler_comparison.py \
  --mode "${MODE}" \
  --data-path "${DATA_PATH}" \
  --num-trials 500 \
  --draws 500 \
  --tune 200 \
  --chains 2 \
  --num-threads "${SLURM_CPUS_PER_TASK:-8}" \
  "$@"
