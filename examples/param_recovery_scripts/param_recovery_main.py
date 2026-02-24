#!/usr/bin/env python
"""
Parameter recovery simulation for TADA and aDDM models.

This script:
1. Generates random parameters from priors
2. Simulates data using those parameters
3. Runs inference with TADA (DDM likelihood) and aDDM (JAX likelihood)
4. Saves results for comparison

Usage:
    python param_recovery_main.py [--sim-id SIM_ID] [--output-dir OUTPUT_DIR]

Example:
    python param_recovery_main.py --sim-id 0 --output-dir ./recovery_results/
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import time
import json
from pathlib import Path
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Setup imports
import pymc as pm
import pytensor.tensor as pt
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import hssm
from hssm.likelihoods import DDM

# Import from efficient_fpt
from efficient_fpt.models import DDModel, piecewise_const_func
from efficient_fpt.utils import get_alternating_mu_array

# JAX implementation
try:
    from efficient_fpt_jax.multi_stage import get_addm_fptd_jax_fast, pad_sacc_array_safely
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("Warning: efficient_fpt_jax not available, JAX inference will be skipped")

# =====================================================================
# Configuration
# =====================================================================

NUM_TRIALS = 3000           # Number of trials to simulate per parameter set
NUM_THREADS = 8            # OpenMP threads
TRUNC_NUM = 6              # JAX series truncation

# Sampling configuration
N_DRAWS = 500              # Posterior draws per chain
N_TUNE = 200               # Tuning steps
N_CHAINS = 2               # Number of parallel chains

# Fixed parameters
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


# =====================================================================
# Helper Functions for PyMC Models
# =====================================================================

def get_mu_padded(mu1, mu2, max_d, L, flag):
    """Symbolic padded alternating mu array."""
    idx = pt.arange(max_d)[None, :]
    parity = (idx + flag[:, None]) % 2
    mu_full = pt.switch(pt.eq(parity, 0), mu1[:, None], mu2[:, None])
    return mu_full * (idx < L[:, None])


def get_mu_tada(mu_data, rt, sacc, L, max_d):
    """Compute time-averaged drift for TADA model."""
    idx = pt.arange(max_d)[None, :]
    Lm1 = (L - 1)[:, None]
    
    dt_mid = sacc[:, 1:] - sacc[:, :-1]
    mask_mid = idx[:, :-1] < Lm1
    
    dt_last = rt[:, None] - sacc
    mask_last = pt.eq(idx, Lm1)
    
    mu_sum = (
        pt.sum(mu_data[:, :-1] * dt_mid * mask_mid, axis=1) +
        pt.sum(mu_data * dt_last * mask_last, axis=1)
    )
    return mu_sum / rt


# =====================================================================
# TADA Model (PyMC with DDM likelihood)
# =====================================================================

def build_tada_model(rt_data: np.ndarray, choice_data: np.ndarray,
                     r1_data: np.ndarray, r2_data: np.ndarray,
                     flag_data: np.ndarray, sacc_data: np.ndarray,
                     length_data: np.ndarray, max_d: int,
                     true_params: Dict[str, float]) -> pm.Model:
    """Build TADA model with DDM likelihood."""
    
    M = float(np.max(rt_data))
    dataset = pd.DataFrame({'rt': rt_data, 'response': choice_data})
    
    with pm.Model() as model:
        rt = pm.Data("rt", rt_data.astype("float64"))
        sacc = pm.Data("sacc", sacc_data.astype("float64"))
        L = pm.Data("L", length_data.astype("int32"))
        flag = pm.Data("flag", flag_data.astype("int32"))
        r1 = pm.Data("r1", r1_data.astype("float64"))
        r2 = pm.Data("r2", r2_data.astype("float64"))
        
        # Priors from build_pymc_model function
        eta = pm.Normal("eta", mu=0.0, sigma=1.0)
        kappa = pm.Gamma("kappa", alpha=2.0, beta=4.0)
        a = pm.Gamma("a", alpha=4.0, beta=2.0)
        
        # b_raw = pm.Beta("b_raw", alpha=2.0, beta=2.0)
        # b = pm.Deterministic("b", b_raw * a / M * 0.95)
        b = 0.0
        
        # x0_raw = pm.Beta("x0_raw", alpha=2.0, beta=2.0)
        # x0 = pm.Deterministic("x0", -a + 2.0 * a * x0_raw)
        z = pm.Uniform("z", lower=0.01, upper=0.99)
        
        # Drift transformations
        mu1 = pm.Deterministic("mu1", kappa * (r1 - eta * r2))
        mu2 = pm.Deterministic("mu2", kappa * (eta * r1 - r2))
        
        mu_padded = get_mu_padded(mu1, mu2, max_d, L, flag)
        mu_tada = pm.Deterministic("mu_glam", get_mu_tada(mu_padded, rt, sacc, L, max_d))
        
        # DDM likelihood
        ddm = DDM("ddm", v=mu_tada, a=a, z=z, t=0, observed=dataset.values)
    
    return model


def run_tada_inference(model: pm.Model, seed: int) -> Dict[str, float]:
    """Run MCMC sampling on TADA model and return posterior means and HDI."""
    try:
        with model:
            trace = pm.sample(
                draws=N_DRAWS,
                tune=N_TUNE,
                chains=N_CHAINS,
                random_seed=seed,
                target_accept=0.9,
                return_inferencedata=True,
                progressbar=False,
                cores=N_CHAINS,
            )
        
        # Extract posterior means and HDI
        import arviz as az
        summary = az.summary(trace, var_names=["kappa", "eta", "a", "z"])
        print(summary)
        results = {}
        results['eta'] = float(summary.loc['eta', 'mean'])
        results['kappa'] = float(summary.loc['kappa', 'mean'])
        results['a'] = float(summary.loc['a', 'mean'])
        results['z'] = float(summary.loc['z', 'mean'])
        
        # Extract HDI values (3% to 97% = 94% HDI)
        hdi = az.hdi(trace, var_names=["kappa", "eta", "a", "z"], hdi_prob=0.94)
        results['eta_hdi_3%'] = float(hdi['eta'].values[0])
        results['eta_hdi_97%'] = float(hdi['eta'].values[1])
        results['kappa_hdi_3%'] = float(hdi['kappa'].values[0])
        results['kappa_hdi_97%'] = float(hdi['kappa'].values[1])
        results['a_hdi_3%'] = float(hdi['a'].values[0])
        results['a_hdi_97%'] = float(hdi['a'].values[1])
        results['z_hdi_3%'] = float(hdi['z'].values[0])
        results['z_hdi_97%'] = float(hdi['z'].values[1])
        # results['x0_raw'] = float(summary.loc['x0_raw', 'mean'])
        
        # Transform x0 back
        results['x0'] = -results['a'] + 2.0 * results['a'] * results['z']
        
        return results
    except Exception as e:
        print(f"Error in TADA inference: {e}")
        return {}


# =====================================================================
# aDDM Model (PyMC with JAX likelihood)
# =====================================================================

def build_addm_jax_model(rt_data: np.ndarray, choice_data: np.ndarray,
                        r1_data: np.ndarray, r2_data: np.ndarray,
                        flag_data: np.ndarray, sacc_data: np.ndarray,
                        length_data: np.ndarray, max_d: int,
                        true_params: Dict[str, float]) -> Tuple[pm.Model, Any]:
    """Build aDDM model with JAX likelihood."""
    
    if not JAX_AVAILABLE:
        return None, None
    
    M = float(np.max(rt_data))
    
    # Setup JAX data
    jax_rt = jnp.array(rt_data)
    jax_choice = jnp.array(choice_data)
    jax_d = jnp.array(length_data)
    jax_r1 = jnp.array(r1_data)
    jax_r2 = jnp.array(r2_data)
    jax_flag = jnp.array(flag_data)
    jax_sacc = jnp.array(sacc_data)
    
    # Pre-compute safe saccade arrays
    jax_sacc_safe = vmap(lambda s, d: pad_sacc_array_safely(s, d, max_d))(jax_sacc, jax_d)
    jax_sacc_safe.block_until_ready()
    
    # Define JAX likelihood functions
    def compute_mu_arrays_jax(eta, kappa, r1, r2, flag, max_d):
        mu1 = kappa * (r1 - eta * r2)
        mu2 = kappa * (eta * r1 - r2)
        indices = jnp.arange(max_d)
        mu_array = jnp.where(
            (indices % 2) == flag,
            mu1,
            mu2
        )
        return mu_array
    
    def jax_loglik_single(rt, choice, d, r1, r2, flag, sacc_safe,
                         eta, kappa, a, b, x0, sigma, max_d, trunc_num):
        mu_array = compute_mu_arrays_jax(eta, kappa, r1, r2, flag, max_d)
        fptd = get_addm_fptd_jax_fast(
            rt, d, mu_array, sacc_safe, sigma, a, b, x0, choice,
            order=30, trunc_num=trunc_num, safe_sacc=sacc_safe
        )
        return jnp.log(jnp.maximum(fptd, 1e-30))
    
    def jax_loglik_batch(eta, kappa, a, b, x0):
        loglik_fn = vmap(
            lambda rt, choice, d, r1, r2, flag, sacc_safe: jax_loglik_single(
                rt, choice, d, r1, r2, flag, sacc_safe,
                eta, kappa, a, b, x0, FIXED_SIGMA, max_d, TRUNC_NUM
            )
        )
        logliks = loglik_fn(jax_rt, jax_choice, jax_d, jax_r1, jax_r2, jax_flag, jax_sacc_safe)
        return jnp.sum(logliks)
    
    jax_loglik_jit = jit(jax_loglik_batch)
    jax_grad_loglik = jit(grad(jax_loglik_batch, argnums=(0, 1, 2, 3, 4)))
    
    # Warm up JIT
    _ = jax_loglik_jit(true_params['eta'], true_params['kappa'],
                      true_params['a'], true_params['b'], true_params['x0'])
    _ = jax_grad_loglik(true_params['eta'], true_params['kappa'],
                       true_params['a'], true_params['b'], true_params['x0'])
    
    # Define PyTensor Ops
    from pytensor.link.jax.dispatch import jax_funcify
    from pytensor.graph.op import Op
    
    class LogLikeJAX(Op):
        itypes = [pt.dvector]
        otypes = [pt.dscalar]
        
        def __init__(self, loglik_fn, grad_fn):
            self.loglik_fn = loglik_fn
            self.grad_fn = grad_fn
        
        def perform(self, node, inputs, outputs):
            (theta,) = inputs
            eta, kappa, a, b, x0 = theta
            loglik = float(self.loglik_fn(eta, kappa, a, b, x0))
            outputs[0][0] = np.array(loglik, dtype="float64")
        
        def grad(self, inputs, output_grads):
            (theta,) = inputs
            (g_out,) = output_grads
            grads = LogLikeJAXGrad(self.grad_fn)(theta)
            return [g_out * grads]
    
    class LogLikeJAXGrad(Op):
        itypes = [pt.dvector]
        otypes = [pt.dvector]
        
        def __init__(self, grad_fn):
            self.grad_fn = grad_fn
        
        def perform(self, node, inputs, outputs):
            (theta,) = inputs
            eta, kappa, a, b, x0 = theta
            grads = self.grad_fn(eta, kappa, a, b, x0)
            grad_array = np.array([float(g) for g in grads], dtype="float64")
            outputs[0][0] = grad_array
    
    @jax_funcify.register(LogLikeJAX)
    def jax_funcify_LogLikeJAX(op, **kwargs):
        loglik_fn = op.loglik_fn
        def loglik_jax(theta):
            eta, kappa, a, b, x0 = theta[0], theta[1], theta[2], theta[3], theta[4]
            return loglik_fn(eta, kappa, a, b, x0)
        return loglik_jax
    
    @jax_funcify.register(LogLikeJAXGrad)
    def jax_funcify_LogLikeJAXGrad(op, **kwargs):
        grad_fn = op.grad_fn
        def grad_jax(theta):
            eta, kappa, a, b, x0 = theta[0], theta[1], theta[2], theta[3], theta[4]
            grads = grad_fn(eta, kappa, a, b, x0)
            return jnp.stack(grads)
        return grad_jax
    
    jax_loglike_op = LogLikeJAX(jax_loglik_jit, jax_grad_loglik)
    
    # Build PyMC model
    with pm.Model() as model:
        eta = pm.Normal("eta", mu=0.0, sigma=1.0)
        kappa = pm.Gamma("kappa", alpha=2.0, beta=4.0)
        a = pm.Gamma("a", alpha=4.0, beta=2.0)
        b = 0.0
        x0_raw = pm.Beta("x0_raw", alpha=2.0, beta=2.0)
        x0 = pm.Deterministic("x0", -a + 2.0 * a * x0_raw)
        
        theta = pt.stack([eta, kappa, a, b, x0])
        pm.Potential("loglik", jax_loglike_op(theta))
    
    return model, jax_loglik_jit


def run_addm_inference(model: pm.Model, seed: int) -> Dict[str, float]:
    """Run MCMC sampling on aDDM model and return posterior means and HDI."""
    if model is None:
        return {}
    
    try:
        with model:
            trace = pm.sample(
                draws=N_DRAWS,
                tune=N_TUNE,
                chains=N_CHAINS,
                nuts_sampler="numpyro",
                random_seed=seed,
                target_accept=0.9,
                progressbar=False,
                return_inferencedata=True,
            )
        
        import arviz as az
        summary = az.summary(trace, var_names=["kappa", "eta", "a", "x0_raw"])
        print(summary)
        results = {}
        results['eta'] = float(summary.loc['eta', 'mean'])
        results['kappa'] = float(summary.loc['kappa', 'mean'])
        results['a'] = float(summary.loc['a', 'mean'])
        results['x0_raw'] = float(summary.loc['x0_raw', 'mean'])
        
        # Extract HDI values (3% to 97% = 94% HDI)
        hdi = az.hdi(trace, var_names=["kappa", "eta", "a", "x0_raw"], hdi_prob=0.94)
        results['eta_hdi_3%'] = float(hdi['eta'].values[0])
        results['eta_hdi_97%'] = float(hdi['eta'].values[1])
        results['kappa_hdi_3%'] = float(hdi['kappa'].values[0])
        results['kappa_hdi_97%'] = float(hdi['kappa'].values[1])
        results['a_hdi_3%'] = float(hdi['a'].values[0])
        results['a_hdi_97%'] = float(hdi['a'].values[1])
        results['x0_raw_hdi_3%'] = float(hdi['x0_raw'].values[0])
        results['x0_raw_hdi_97%'] = float(hdi['x0_raw'].values[1])
        
        # Transform x0 back
        results['x0'] = -results['a'] + 2.0 * results['a'] * results['x0_raw']
        
        return results
    except Exception as e:
        print(f"Error in aDDM inference: {e}")
        return {}


# =====================================================================
# Main Parameter Recovery Simulation
# =====================================================================

def run_single_simulation(sim_id: int, output_dir: Path, seed: int,
                         simulate_eta: bool, eta_const: float,
                         simulate_kappa: bool, kappa_const: float,
                         simulate_a: bool, a_const: float,
                         simulate_x0: bool, x0_const: float) -> Dict[str, Any]:
    """Run a single parameter recovery simulation.
    
    Args:
        sim_id: Simulation ID
        output_dir: Output directory path
        seed: Random seed
        simulate_eta: Whether to sample eta from prior
        eta_const: Constant eta value if not sampling
        simulate_kappa: Whether to sample kappa from prior
        kappa_const: Constant kappa value if not sampling
        simulate_a: Whether to sample a from prior
        a_const: Constant a value if not sampling
        simulate_x0: Whether to sample x0 from prior
        x0_const: Constant x0 value if not sampling (None = derived from a)
    """
    
    print(f"\n{'='*70}")
    print(f"Simulation {sim_id}")
    print(f"{'='*70}")
    
    # Generate random true parameters
    rng = np.random.default_rng(seed)
    
    true_params = {'sigma': FIXED_SIGMA}
    
    # Sample or use constant for eta
    if simulate_eta:
        true_params['eta'] = ParameterPriors.sample_eta(rng)
    else:
        true_params['eta'] = eta_const
    
    # Sample or use constant for kappa
    if simulate_kappa:
        true_params['kappa'] = ParameterPriors.sample_kappa(rng)
    else:
        true_params['kappa'] = kappa_const
    
    # Sample or use constant for a
    if simulate_a:
        true_params['a'] = ParameterPriors.sample_a(rng)
    else:
        true_params['a'] = a_const
    
    # b is always 0 (no collapsing boundary)
    true_params['b'] = 0.0
    
    # Sample or use constant for x0
    if simulate_x0:
        true_params['x0'] = ParameterPriors.sample_x0(rng, true_params['a'])
    else:
        if x0_const is not None:
            true_params['x0'] = x0_const
        else:
            # Default: sample x0 even if not explicitly requested
            true_params['x0'] = ParameterPriors.sample_x0(rng, true_params['a'])
    # true_params['x0'] = 0.5
    
    print(f"True Parameters:")
    for k, v in true_params.items():
        print(f"  {k}: {v:.6f}")
    
    # Simulate dataset
    print(f"\nSimulating {NUM_TRIALS} trials...")

    data_start = time.time()
    
    (rt_data, choice_data, d_data, r1_data, r2_data, flag_data,
     mu_data_padded, sacc_data_padded, mu1_data, mu2_data, length_data, max_d) = \
        simulate_dataset(true_params, NUM_TRIALS, seed * 1000 + sim_id)
    
    data_time = time.time() - data_start
    print(f"Data simulation completed in {data_time:.1f}s")
    print(f"  Trials: {len(rt_data)}, Max stages: {max_d}")
    print(f"  RT range: [{rt_data.min():.4f}, {rt_data.max():.4f}]")
    print(f"  Upper choices: {np.sum(choice_data == 1)} ({100*np.mean(choice_data == 1):.1f}%)")
    
    # Save simulated parameters immediately after generation
    save_simulated_data(sim_id, true_params, rt_data, choice_data, d_data,
                       r1_data, r2_data, flag_data, mu_data_padded,
                       sacc_data_padded, mu1_data, mu2_data, length_data, output_dir)
    
    results = {
        'sim_id': sim_id,
        'true_params': true_params,
        'tada_posterior': {},
        'addm_posterior': {},
        'timing': {
            'data_simulation': data_time,
            'tada_inference': 0.0,
            'addm_inference': 0.0,
        }
    }
    
    # Run TADA inference
    print(f"\nRunning TADA inference (DDM likelihood)...")
    tada_start = time.time()
    try:
        tada_model = build_tada_model(rt_data, choice_data, r1_data, r2_data,
                                      flag_data, sacc_data_padded, length_data,
                                      max_d, true_params)
        tada_results = run_tada_inference(tada_model, seed * 10000 + sim_id)
            
            # Transform z HDI to x0 HDI
        if 'z_hdi_3%' in tada_results and 'z_hdi_97%' in tada_results:
                a_mean = tada_results['a']
                z_hdi_3 = tada_results['z_hdi_3%']
                z_hdi_97 = tada_results['z_hdi_97%']
                x0_hdi_3 = -a_mean + 2.0 * a_mean * z_hdi_3
                x0_hdi_97 = -a_mean + 2.0 * a_mean * z_hdi_97
                tada_results['x0_hdi_3%'] = x0_hdi_3
                tada_results['x0_hdi_97%'] = x0_hdi_97
    
        results['tada_posterior'] = tada_results
        results['timing']['tada_inference'] = time.time() - tada_start
        print(f"TADA inference completed in {results['timing']['tada_inference']:.1f}s")
        print(f"TADA Posterior Means: {tada_results}")
    except Exception as e: 
        print(f"TADA inference failed: {e}")
        results['tada_posterior'] = {}
    
    # Run aDDM inference
    if JAX_AVAILABLE:
        print(f"\nRunning aDDM inference (JAX likelihood)...")
        addm_start = time.time()
        try:
            addm_model, _ = build_addm_jax_model(rt_data, choice_data, r1_data, r2_data,
                                                flag_data, sacc_data_padded, length_data,
                                                max_d, true_params)
            addm_results = run_addm_inference(addm_model, seed * 20000 + sim_id)
            
            # Transform x0_raw HDI to x0 HDI
            if 'x0_raw_hdi_3%' in addm_results and 'x0_raw_hdi_97%' in addm_results:
                a_mean = addm_results['a']
                x0_raw_hdi_3 = addm_results['x0_raw_hdi_3%']
                x0_raw_hdi_97 = addm_results['x0_raw_hdi_97%']
                x0_hdi_3 = -a_mean + 2.0 * a_mean * x0_raw_hdi_3
                x0_hdi_97 = -a_mean + 2.0 * a_mean * x0_raw_hdi_97
                addm_results['x0_hdi_3%'] = x0_hdi_3
                addm_results['x0_hdi_97%'] = x0_hdi_97
            
            results['addm_posterior'] = addm_results
            results['timing']['addm_inference'] = time.time() - addm_start
            print(f"aDDM inference completed in {results['timing']['addm_inference']:.1f}s")
            print(f"aDDM Posterior Means: {addm_results}")
        except Exception as e:
            print(f"aDDM inference failed: {e}")
            results['addm_posterior'] = {}
    
    return results


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


def save_results(results: Dict[str, Any], output_dir: Path) -> None:
    """Save results to a structured format."""
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual result as JSON
    sim_id = results['sim_id']
    json_path = output_dir / f"sim_{sim_id:05d}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {
        'sim_id': results['sim_id'],
        'true_params': {k: float(v) for k, v in results['true_params'].items()},
        'tada_posterior': {k: float(v) for k, v in results['tada_posterior'].items()},
        'addm_posterior': {k: float(v) for k, v in results['addm_posterior'].items()},
        'timing': results['timing'],
    }
    
    with open(json_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Parameter recovery simulation")
    parser.add_argument('--sim-id', type=int, default=0,
                       help='Simulation ID (default: 0)')
    parser.add_argument('--output-dir', type=str, default='./recovery_results/',
                       help='Output directory (default: ./recovery_results/)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed (default: 42)')
    parser.add_argument('--simulate-eta', type=str, default='FALSE',
                       help='Sample eta from prior (TRUE/FALSE)')
    parser.add_argument('--eta-const', type=float, default=0.3,
                       help='Constant eta value when not sampling (default: 0.3)')
    parser.add_argument('--simulate-kappa', type=str, default='FALSE',
                       help='Sample kappa from prior (TRUE/FALSE)')
    parser.add_argument('--kappa-const', type=float, default=0.5,
                       help='Constant kappa value when not sampling (default: 0.5)')
    parser.add_argument('--simulate-a', type=str, default='FALSE',
                       help='Sample a from prior (TRUE/FALSE)')
    parser.add_argument('--a-const', type=float, default=2.0,
                       help='Constant a value when not sampling (default: 2.0)')
    parser.add_argument('--simulate-x0', type=str, default='FALSE',
                       help='Sample x0 from prior (TRUE/FALSE)')
    parser.add_argument('--x0-const', type=float, default=None,
                       help='Constant x0 value when not sampling (default: None, derived from a)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    seed = args.seed + args.sim_id
    
    # Convert string arguments to boolean
    simulate_eta = args.simulate_eta.upper() == 'TRUE'
    simulate_kappa = args.simulate_kappa.upper() == 'TRUE'
    simulate_a = args.simulate_a.upper() == 'TRUE'
    simulate_x0 = args.simulate_x0.upper() == 'TRUE'
    
    print(f"\nParameter Sampling Configuration:")
    print(f"  eta: {'sample from prior' if simulate_eta else f'constant ({args.eta_const})'}")
    print(f"  kappa: {'sample from prior' if simulate_kappa else f'constant ({args.kappa_const})'}")
    print(f"  a: {'sample from prior' if simulate_a else f'constant ({args.a_const})'}")
    print(f"  x0: {'sample from prior' if simulate_x0 else f'constant ({args.x0_const})'}")
    
    # Run simulation
    results = run_single_simulation(
        args.sim_id, output_dir, seed,
        simulate_eta, args.eta_const,
        simulate_kappa, args.kappa_const,
        simulate_a, args.a_const,
        simulate_x0, args.x0_const
    )
    
    # Save results
    save_results(results, output_dir)
    
    print(f"\n{'='*70}")
    print(f"Simulation {args.sim_id} completed successfully!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
