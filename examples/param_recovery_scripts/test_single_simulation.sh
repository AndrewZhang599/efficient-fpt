#!/bin/bash
# Test script to run a single parameter recovery simulation locally
# Useful for debugging or testing before submitting 500 jobs

set -e  # Exit on error

echo "======================================================================"
echo "Parameter Recovery Test - Running Single Simulation"
echo "======================================================================"
echo ""

# Configuration
SIM_ID=${1:-0}
OUTPUT_DIR="/users/azhan378/scratch/param_recovery"
SEED=42

echo "Running test simulation..."
echo "  Simulation ID: $SIM_ID"
echo "  Output directory: $OUTPUT_DIR"
echo "  Random seed: $SEED"
echo ""

# # Activate virtual environment
# echo "Activating virtual environment..."
# source /users/azhan378/hssm_oscar_uv/bin/activate

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run single simulation
echo "Starting parameter recovery simulation..."
python param_recovery_main.py \
    --sim-id "$SIM_ID" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED"

echo ""
echo "======================================================================"
echo "Test Complete!"
echo "======================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR/sim_$(printf "%05d" $SIM_ID).json"
echo ""
echo "To view results:"
echo "  python -c \"import json; r=json.load(open('$OUTPUT_DIR/sim_$(printf "%05d" $SIM_ID).json')); print('True eta:', r['true_params']['eta']); print('TADA eta:', r['tada_posterior'].get('eta', 'FAILED')); print('aDDM eta:', r['addm_posterior'].get('eta', 'FAILED'))\""
echo ""
echo "If this worked, you can submit the full array:"
echo "  sbatch param_recovery.sbatch"
