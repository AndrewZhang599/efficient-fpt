#!/usr/bin/env python
"""
Data generation script for parameter recovery simulations.

This script generates synthetic data from the aDDM model with random parameters.
Generated data is saved to individual folders per simulation for later inference.

Usage:
    python data_generation.py --sim-id 0 --output-dir ./recovery_results/

Output structure:
    output_dir/simulated_params/sim_00000/
        ├── true_params.json
        └── simulated_data.npz
"""

import os
import sys
import argparse
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import from efficient_fpt
from efficient_fpt.models import DDModel, piecewise_const_func
from efficient_fpt.utils import get_alternating_mu_array

# =====================================================================
# Configuration
# =====================================================================

NUM_TRIALS = 3000           # Number of trials to simulate per parameter set
FIXED_SIGMA = 1.0          # Diffusion coefficient (always fixed at 1.0)
FIXATION_SHAPE = 6         # Gamma shape for fixation times
FIXATION_SCALE = 0.1       # Gamma scale for fixation times

# =====================================================================
# Prior Definitions
# =====================================================================

class ParameterPriors:
    """Helper class for sampling parameters from priors."""
    
    @staticmethod
    def sample_eta(rng: np.random.Generator) -> float:
        """Sample eta from Beta(2,2)"""
        return rng.beta(a=2.0, b=2.0)
    
    @staticmethod
    def sample_kappa(rng: np.random.Generator) -> float:
        """Sample kappa from Gamma(2, 4)."""
        return rng.gamma(shape=2.0, scale=1.0/4.0)
    
    @staticmethod
    def sample_a(rng: np.random.Generator) -> float:
        """Sample a (boundary) from Gamma(4, 2)."""
        return rng.gamma(shape=4.0, scale=1.0/2.0)
    
    @staticmethod
    def sample_b(rng: np.random.Generator, a: float, t_max: float) -> float:
        """Sample b (collapse rate) from Beta, scaled appropriately.
        
        Constraint: a - b*t_max > 0, so b < a/t_max
        """
        b_raw = rng.beta(a=2.0, b=2.0)
        b = b_raw * a / t_max * 0.95  # 0.95 for safety margin
        return b
    
    @staticmethod
    def sample_x0(rng: np.random.Generator, a: float) -> float:
        """Sample x0 from Beta(2, 2), transformed to [-a, a]."""
        x0_raw = rng.beta(a=2.0, b=2.0)
        x0 = -a + 2.0 * a * x0_raw
        return x0


# =====================================================================
# Data Simulation
# =====================================================================

class aDDModel(DDModel):
    """Attentional Drift Diffusion Model for a single trial."""
    def __init__(self, mu1, mu2, sacc_array, flag, sigma, a, b, x0):
        super().__init__(x0)
        self.mu1 = mu1
        self.mu2 = mu2
        self.sacc_array = sacc_array
        self.flag = flag
        self.d = len(sacc_array)
        self.mu_array = get_alternating_mu_array(mu1, mu2, self.d, flag)
        self.sigma = sigma
        self.a = a
        self.b = b

    def drift_coeff(self, X, t):
        return piecewise_const_func(t, self.mu_array, self.sacc_array)

    def diffusion_coeff(self, X, t):
        return self.sigma

    @property
    def is_update_vectorizable(self):
        return True

    def upper_bdy(self, t):
        return self.a - self.b * t

    def lower_bdy(self, t):
        return -self.a + self.b * t


def simulate_trial(rng: np.random.Generator, eta: float, kappa: float, 
                   sigma: float, a: float, b: float, x0: float,
                   t_max: float, shape: float, scale: float) -> Dict[str, Any]:
    """Simulate a single aDDM trial."""
    # Generate fixation times (saccade array)
    fixations = rng.gamma(shape, scale, 100)
    sacc_array = np.insert(np.cumsum(fixations), 0, 0)
    sacc_array = sacc_array[sacc_array < t_max]
    
    # Random initial attention and stimulus values
    flag = rng.integers(0, 2)
    r1 = rng.integers(1, 6)
    r2 = rng.integers(1, 6)
    
    # Compute drift rates
    mu1 = kappa * (r1 - eta * r2)
    mu2 = kappa * (eta * r1 - r2)
    
    # Simulate
    model = aDDModel(mu1, mu2, sacc_array, flag, sigma, a, b, x0)
    rt, choice = model.simulate_fpt_datum(dt=1e-4)
    
    # Truncate saccade array to actual RT
    sacc_array = sacc_array[sacc_array < rt]
    d = len(sacc_array)
    mu_array = get_alternating_mu_array(mu1, mu2, d, flag)
    
    return {
        'rt': rt,
        'choice': int(choice),
        'mu_array': mu_array,
        'sacc_array': sacc_array,
        'd': d,
        'r1': r1,
        'r2': r2,
        'flag': flag,
        'mu1': mu1,
        'mu2': mu2,
    }


def simulate_dataset(true_params: Dict[str, float], num_trials: int, 
                    seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                        np.ndarray, np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray, int]:
    """Simulate a complete dataset with given parameters."""
    rng = np.random.default_rng(seed)
    t_max = true_params['a']
    
    trials = []
    for i in range(num_trials):
        trial = simulate_trial(
            rng, true_params['eta'], true_params['kappa'],
            true_params['sigma'], true_params['a'], true_params['b'],
            true_params['x0'], t_max, FIXATION_SHAPE, FIXATION_SCALE
        )
        trials.append(trial)
    
    # Extract and pad arrays
    rt_data = np.array([t['rt'] for t in trials], dtype=np.float64)
    choice_data = np.array([t['choice'] for t in trials], dtype=np.int32)
    d_data = np.array([t['d'] for t in trials], dtype=np.int32)
    r1_data = np.array([t['r1'] for t in trials], dtype=np.float64)
    r2_data = np.array([t['r2'] for t in trials], dtype=np.float64)
    flag_data = np.array([t['flag'] for t in trials], dtype=np.int32)
    mu1_data = np.array([t['mu1'] for t in trials], dtype=np.float64)
    mu2_data = np.array([t['mu2'] for t in trials], dtype=np.float64)
    
    max_d = max(d_data)
    mu_data_padded = np.zeros((num_trials, max_d), dtype=np.float64)
    sacc_data_padded = np.zeros((num_trials, max_d), dtype=np.float64)
    
    for i, t in enumerate(trials):
        d = t['d']
        mu_data_padded[i, :d] = t['mu_array']
        sacc_data_padded[i, :d] = t['sacc_array']
    
    return (rt_data, choice_data, d_data, r1_data, r2_data, flag_data,
            mu_data_padded, sacc_data_padded, mu1_data, mu2_data, d_data, max_d)


def save_simulated_data(sim_id: int, true_params: Dict[str, float], 
                       rt_data: np.ndarray, choice_data: np.ndarray,
                       d_data: np.ndarray, r1_data: np.ndarray, r2_data: np.ndarray,
                       flag_data: np.ndarray, mu_data_padded: np.ndarray,
                       sacc_data_padded: np.ndarray, mu1_data: np.ndarray,
                       mu2_data: np.ndarray, length_data: np.ndarray,
                       output_dir: Path) -> None:
    """Save simulated parameter sets in a separate folder with sim_id."""
    
    # Create simulated_params subdirectory with sim_id
    sim_params_dir = output_dir / "simulated_params" / f"sim_{sim_id:05d}"
    sim_params_dir.mkdir(parents=True, exist_ok=True)
    
    # Save true parameters as JSON
    true_params_serializable = {k: float(v) for k, v in true_params.items()}
    true_params_path = sim_params_dir / "true_params.json"
    with open(true_params_path, 'w') as f:
        json.dump(true_params_serializable, f, indent=2)
    
    # Save simulated data as numpy arrays
    data = {
        'rt_data': rt_data,
        'choice_data': choice_data,
        'd_data': d_data,
        'r1_data': r1_data,
        'r2_data': r2_data,
        'flag_data': flag_data,
        'mu_data_padded': mu_data_padded,
        'sacc_data_padded': sacc_data_padded,
        'mu1_data': mu1_data,
        'mu2_data': mu2_data,
        'length_data': length_data,
    }
    
    data_path = sim_params_dir / "simulated_data.npz"
    np.savez(data_path, **data)
    
    print(f"Simulated parameters saved to {sim_params_dir}/")


def run_single_data_generation(sim_id: int, output_dir: Path, seed: int) -> None:
    """Generate and save synthetic data for a single simulation."""
    
    print(f"\n{'='*70}")
    print(f"Data Generation - Simulation {sim_id}")
    print(f"{'='*70}")
    
    # Generate random true parameters
    rng = np.random.default_rng(seed)
    
    # true_params = {
    #     'eta': ParameterPriors.sample_eta(rng),
    #     'kappa': ParameterPriors.sample_kappa(rng),
    #     'a': ParameterPriors.sample_a(rng),
    #     'sigma': FIXED_SIGMA,
    # }
    true_params = {
        'eta': 0.3, 
        'kappa': ParameterPriors.sample_kappa(rng),
        'a': 2.0,
        'sigma': FIXED_SIGMA,
    }
    
    # Sample b and x0 with constraints
    true_params['b'] = 0.0
    # true_params['x0'] = ParameterPriors.sample_x0(rng, true_params['a'])
    true_params['x0'] = 0.5
    
    print(f"True Parameters:")
    for k, v in true_params.items():
        print(f"  {k}: {v:.6f}")
    
    # Simulate dataset
    print(f"\nSimulating {NUM_TRIALS} trials...")
    
    (rt_data, choice_data, d_data, r1_data, r2_data, flag_data,
     mu_data_padded, sacc_data_padded, mu1_data, mu2_data, length_data, max_d) = \
        simulate_dataset(true_params, NUM_TRIALS, seed * 1000 + sim_id)
    
    print(f"Data simulation completed")
    print(f"  Trials: {len(rt_data)}, Max stages: {max_d}")
    print(f"  RT range: [{rt_data.min():.4f}, {rt_data.max():.4f}]")
    print(f"  Upper choices: {np.sum(choice_data == 1)} ({100*np.mean(choice_data == 1):.1f}%)")
    
    # Save simulated parameters and data
    output_dir_path = Path(output_dir)
    save_simulated_data(sim_id, true_params, rt_data, choice_data, d_data,
                       r1_data, r2_data, flag_data, mu_data_padded,
                       sacc_data_padded, mu1_data, mu2_data, length_data, output_dir_path)
    
    print(f"\n{'='*70}")
    print(f"Data generation {sim_id} completed successfully!")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data for parameter recovery")
    parser.add_argument('--sim-id', type=int, default=0,
                       help='Simulation ID (default: 0)')
    parser.add_argument('--output-dir', type=str, default='./recovery_results/',
                       help='Output directory (default: ./recovery_results/)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed (default: 42)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    seed = args.seed + args.sim_id
    
    # Run data generation
    run_single_data_generation(args.sim_id, output_dir, seed)


if __name__ == '__main__':
    main()
