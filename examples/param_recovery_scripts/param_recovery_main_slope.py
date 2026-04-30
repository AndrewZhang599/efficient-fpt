#!/usr/bin/env python
"""
Parameter recovery simulation comparing the slope parameter for the aDDM
(b in a - b*t) against the HSSM Angle model (theta, in radians, where the
boundary is the constant boundary minus a slope of tan(theta)).

This script:
1. Generates random parameters from priors (now including the slope b)
2. Simulates data using the FAST Cython aDDM simulator (`simulate_addm`)
3. Runs inference with:
     - HSSM Angle model (theta = arctan(b)) on time-averaged drift (TADA)
     - aDDM (JAX likelihood) recovering eta, kappa, a, b, x0
4. Saves results for slope-recovery comparison

Conversion conventions
----------------------
- aDDM:  upper boundary is `a - b*t`, so `b` is the slope of decrease.
- Angle: HSSM angle model parameterizes the slope by `theta` (radians),
  where the slope of the boundary is `tan(theta)`. HSSM constrains theta
  to roughly [-0.1, 1.3] rad.
- For an apples-to-apples comparison, `b = tan(theta)`. We sample `b` from
  Beta(2, 2) (so b in [0, 1]) for both models, which corresponds to
  theta = arctan(b) in [0, pi/4] (well within HSSM's valid range).

Usage:
    python param_recovery_main_slope.py [--sim-id SIM_ID] [--output-dir OUTPUT_DIR]
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
from hssm.distribution_utils import make_distribution_for_supported_model

# Fast Cython aDDM simulator
from efficient_fpt.addm import simulate_addm

# JAX likelihood for inference
try:
    from efficient_fpt_jax.multi_stage import get_addm_fptd_jax_fast, pad_sacc_array_safely
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("Warning: efficient_fpt_jax not available, JAX inference will be skipped")

# HSSM Angle distribution (built once at import; reused per-model)
ANGLE = make_distribution_for_supported_model("angle", loglik_kind="approx_differentiable")

# =====================================================================
# Configuration
# =====================================================================

NUM_TRIALS = 1000          # Number of trials to simulate per parameter set
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

SIM_DT = 1e-4              # Simulation timestep (Cython simulator)


# =====================================================================
# Prior Definitions
# =====================================================================

class ParameterPriors:
    """Helper class for sampling parameters from priors."""

    @staticmethod
    def sample_eta(rng: np.random.Generator) -> float:
        """Sample eta from Beta(2,2)."""
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
    def sample_b(rng: np.random.Generator) -> float:
        """Sample b (collapse rate) from Beta(2, 2). b in [0, 1]."""
        return rng.beta(a=2.0, b=2.0)

    @staticmethod
    def sample_x0(rng: np.random.Generator, a: float) -> float:
        """Sample x0 from Beta(2, 2), transformed to [-a, a]."""
        x0_raw = rng.beta(a=2.0, b=2.0)
        return -a + 2.0 * a * x0_raw


# =====================================================================
# Data Simulation (Cython-backed)
# =====================================================================

def simulate_dataset(true_params: Dict[str, float], num_trials: int,
                     seed: int) -> Tuple[np.ndarray, ...]:
    """Simulate a complete dataset using the fast Cython aDDM simulator.

    Returns the same tuple shape as the original Python script.
    """
    a = true_params['a']
    b = true_params['b']
    # Ensure max_t is comfortably above the boundary-collapse time (a/b)
    # so trials can terminate, while still leaving room for fixations.
    if b > 0.0:
        max_t = max(10.0, a / b)
    else:
        max_t = 10.0

    out = simulate_addm(
        n_trials=num_trials,
        eta=true_params['eta'],
        kappa=true_params['kappa'],
        sigma=true_params['sigma'],
        a=a,
        b=b,
        x0=true_params['x0'],
        gamma_shape=FIXATION_SHAPE,
        gamma_scale=FIXATION_SCALE,
        r_range=(1, 5),
        dt=SIM_DT,
        max_t=max_t,
        n_threads=1,
        random_state=seed,
    )

    rt_data = out['rt'].astype(np.float64)
    choice_data = out['choice'].astype(np.int32)
    d_data = out['d_data'].astype(np.int32)
    r1_data = out['r1'].astype(np.float64)
    r2_data = out['r2'].astype(np.float64)
    flag_data = out['flag'].astype(np.int32)
    mu_data_padded = out['mu_data_padded'].astype(np.float64)
    sacc_data_padded = out['sacc_data_padded'].astype(np.float64)
    mu1_data = out['mu1'].astype(np.float64)
    mu2_data = out['mu2'].astype(np.float64)

    # Drop any trials that did not terminate (rt <= 0). These are very rare
    # when max_t is set sensibly but would break the likelihoods.
    terminated = rt_data > 0
    if not np.all(terminated):
        keep = np.where(terminated)[0]
        rt_data = rt_data[keep]
        choice_data = choice_data[keep]
        d_data = d_data[keep]
        r1_data = r1_data[keep]
        r2_data = r2_data[keep]
        flag_data = flag_data[keep]
        mu_data_padded = mu_data_padded[keep]
        sacc_data_padded = sacc_data_padded[keep]
        mu1_data = mu1_data[keep]
        mu2_data = mu2_data[keep]

    max_d = int(d_data.max())
    # Trim padded columns to actual max_d in case simulator over-allocated
    mu_data_padded = np.ascontiguousarray(mu_data_padded[:, :max_d])
    sacc_data_padded = np.ascontiguousarray(sacc_data_padded[:, :max_d])

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
    """Compute time-averaged drift for TADA / Angle model."""
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
# Angle Model (PyMC with HSSM Angle likelihood, TADA drift)
# =====================================================================

def build_angle_model(rt_data: np.ndarray, choice_data: np.ndarray,
                      r1_data: np.ndarray, r2_data: np.ndarray,
                      flag_data: np.ndarray, sacc_data: np.ndarray,
                      length_data: np.ndarray, max_d: int,
                      true_params: Dict[str, float]) -> pm.Model:
    """Build HSSM Angle model with time-averaged drift (TADA)."""

    dataset = pd.DataFrame({'rt': rt_data, 'response': choice_data})

    with pm.Model() as model:
        rt = pm.Data("rt", rt_data.astype("float64"))
        sacc = pm.Data("sacc", sacc_data.astype("float64"))
        L = pm.Data("L", length_data.astype("int32"))
        flag = pm.Data("flag", flag_data.astype("int32"))
        r1 = pm.Data("r1", r1_data.astype("float64"))
        r2 = pm.Data("r2", r2_data.astype("float64"))

        # Priors (matched to the aDDM JAX model so comparisons are fair)
        eta = pm.Normal("eta", mu=0.0, sigma=1.0)
        kappa = pm.Gamma("kappa", alpha=2.0, beta=4.0)
        a = pm.Gamma("a", alpha=4.0, beta=2.0)
        z = pm.Uniform("z", lower=0.01, upper=0.99)

        # Slope: sample b in [0, 1] then convert to theta = arctan(b)
        # so the implied prior on `b` matches the aDDM Beta(2,2) prior.
        b_raw = pm.Beta("b_raw", alpha=2.0, beta=2.0)
        b_angle = pm.Deterministic("b", b_raw)  # equivalent slope
        theta = pm.Deterministic("theta", pt.arctan(b_raw))

        # Time-averaged drift across fixations
        mu1 = pm.Deterministic("mu1", kappa * (r1 - eta * r2))
        mu2 = pm.Deterministic("mu2", kappa * (eta * r1 - r2))
        mu_padded = get_mu_padded(mu1, mu2, max_d, L, flag)
        mu_tada = pm.Deterministic("mu_glam", get_mu_tada(mu_padded, rt, sacc, L, max_d))

        # HSSM Angle likelihood (constant non-decision time fixed at 0)
        ANGLE("angle", v=mu_tada, a=a, z=z, t=0.0,
              theta=theta, observed=dataset.values)

    return model


def run_angle_inference(model: pm.Model, seed: int) -> Dict[str, float]:
    """Run MCMC sampling on Angle model and return posterior means and HDI."""
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

        import arviz as az
        var_names = ["kappa", "eta", "a", "z", "b", "theta"]
        summary = az.summary(trace, var_names=var_names)
        print(summary)
        results = {name: float(summary.loc[name, 'mean']) for name in var_names}

        hdi = az.hdi(trace, var_names=var_names, hdi_prob=0.94)
        for name in var_names:
            results[f'{name}_hdi_3%'] = float(hdi[name].values[0])
            results[f'{name}_hdi_97%'] = float(hdi[name].values[1])

        # Transform z back to x0 in [-a, a] for comparison
        results['x0'] = -results['a'] + 2.0 * results['a'] * results['z']
        results['x0_hdi_3%'] = -results['a'] + 2.0 * results['a'] * results['z_hdi_3%']
        results['x0_hdi_97%'] = -results['a'] + 2.0 * results['a'] * results['z_hdi_97%']

        return results
    except Exception as e:
        print(f"Error in Angle inference: {e}")
        return {}


# =====================================================================
# aDDM Model (PyMC with JAX likelihood) -- now recovers b too
# =====================================================================

def build_addm_jax_model(rt_data: np.ndarray, choice_data: np.ndarray,
                         r1_data: np.ndarray, r2_data: np.ndarray,
                         flag_data: np.ndarray, sacc_data: np.ndarray,
                         length_data: np.ndarray, max_d: int,
                         true_params: Dict[str, float]) -> Tuple[pm.Model, Any]:
    """Build aDDM model with JAX likelihood. Recovers eta, kappa, a, b, x0."""

    if not JAX_AVAILABLE:
        return None, None

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
        return jnp.where((indices % 2) == flag, mu1, mu2)

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

    # Warm up JIT at the true values
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
            (theta_vec,) = inputs
            eta, kappa, a, b, x0 = theta_vec
            loglik = float(self.loglik_fn(eta, kappa, a, b, x0))
            outputs[0][0] = np.array(loglik, dtype="float64")

        def grad(self, inputs, output_grads):
            (theta_vec,) = inputs
            (g_out,) = output_grads
            grads = LogLikeJAXGrad(self.grad_fn)(theta_vec)
            return [g_out * grads]

    class LogLikeJAXGrad(Op):
        itypes = [pt.dvector]
        otypes = [pt.dvector]

        def __init__(self, grad_fn):
            self.grad_fn = grad_fn

        def perform(self, node, inputs, outputs):
            (theta_vec,) = inputs
            eta, kappa, a, b, x0 = theta_vec
            grads = self.grad_fn(eta, kappa, a, b, x0)
            outputs[0][0] = np.array([float(g) for g in grads], dtype="float64")

    @jax_funcify.register(LogLikeJAX)
    def jax_funcify_LogLikeJAX(op, **kwargs):
        loglik_fn = op.loglik_fn
        def loglik_jax(theta_vec):
            eta, kappa, a, b, x0 = theta_vec[0], theta_vec[1], theta_vec[2], theta_vec[3], theta_vec[4]
            return loglik_fn(eta, kappa, a, b, x0)
        return loglik_jax

    @jax_funcify.register(LogLikeJAXGrad)
    def jax_funcify_LogLikeJAXGrad(op, **kwargs):
        grad_fn = op.grad_fn
        def grad_jax(theta_vec):
            eta, kappa, a, b, x0 = theta_vec[0], theta_vec[1], theta_vec[2], theta_vec[3], theta_vec[4]
            grads = grad_fn(eta, kappa, a, b, x0)
            return jnp.stack(grads)
        return grad_jax

    jax_loglike_op = LogLikeJAX(jax_loglik_jit, jax_grad_loglik)

    # Build PyMC model -- now b is sampled (Beta(2,2), in [0,1])
    with pm.Model() as model:
        eta = pm.Normal("eta", mu=0.0, sigma=1.0)
        kappa = pm.Gamma("kappa", alpha=2.0, beta=4.0)
        a = pm.Gamma("a", alpha=4.0, beta=2.0)
        b = pm.Beta("b", alpha=2.0, beta=2.0)
        x0_raw = pm.Beta("x0_raw", alpha=2.0, beta=2.0)
        x0 = pm.Deterministic("x0", -a + 2.0 * a * x0_raw)
        # Equivalent angle theta = arctan(b), for direct comparison with HSSM Angle
        theta_eq = pm.Deterministic("theta", pt.arctan(b))

        theta_vec = pt.stack([eta, kappa, a, b, x0])
        pm.Potential("loglik", jax_loglike_op(theta_vec))

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
        var_names = ["kappa", "eta", "a", "b", "x0_raw", "theta"]
        summary = az.summary(trace, var_names=var_names)
        print(summary)
        results = {name: float(summary.loc[name, 'mean']) for name in var_names}

        hdi = az.hdi(trace, var_names=var_names, hdi_prob=0.94)
        for name in var_names:
            results[f'{name}_hdi_3%'] = float(hdi[name].values[0])
            results[f'{name}_hdi_97%'] = float(hdi[name].values[1])

        # Transform x0_raw to x0
        results['x0'] = -results['a'] + 2.0 * results['a'] * results['x0_raw']
        results['x0_hdi_3%'] = -results['a'] + 2.0 * results['a'] * results['x0_raw_hdi_3%']
        results['x0_hdi_97%'] = -results['a'] + 2.0 * results['a'] * results['x0_raw_hdi_97%']

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
                          simulate_b: bool, b_const: float,
                          simulate_x0: bool, x0_const: float) -> Dict[str, Any]:
    """Run a single parameter recovery simulation."""

    print(f"\n{'='*70}")
    print(f"Simulation {sim_id}")
    print(f"{'='*70}")

    rng = np.random.default_rng(seed)
    true_params = {'sigma': FIXED_SIGMA}

    if simulate_eta:
        true_params['eta'] = ParameterPriors.sample_eta(rng)
    else:
        true_params['eta'] = eta_const

    if simulate_kappa:
        true_params['kappa'] = ParameterPriors.sample_kappa(rng)
    else:
        true_params['kappa'] = kappa_const

    if simulate_a:
        true_params['a'] = ParameterPriors.sample_a(rng)
    else:
        true_params['a'] = a_const

    # b is now sampled (or fixed) -- in [0, 1] via Beta(2,2)
    if simulate_b:
        true_params['b'] = ParameterPriors.sample_b(rng)
    else:
        true_params['b'] = b_const

    # Equivalent angle from b for record-keeping
    true_params['theta'] = float(np.arctan(true_params['b']))

    if simulate_x0:
        true_params['x0'] = ParameterPriors.sample_x0(rng, true_params['a'])
    else:
        if x0_const is not None:
            true_params['x0'] = x0_const
        else:
            true_params['x0'] = ParameterPriors.sample_x0(rng, true_params['a'])

    print(f"True Parameters:")
    for k, v in true_params.items():
        print(f"  {k}: {v:.6f}")

    # Simulate dataset
    print(f"\nSimulating {NUM_TRIALS} trials (Cython aDDM simulator)...")
    data_start = time.time()
    (rt_data, choice_data, d_data, r1_data, r2_data, flag_data,
     mu_data_padded, sacc_data_padded, mu1_data, mu2_data, length_data, max_d) = \
        simulate_dataset(true_params, NUM_TRIALS, seed * 1000 + sim_id)
    data_time = time.time() - data_start
    print(f"Data simulation completed in {data_time:.1f}s")
    print(f"  Trials: {len(rt_data)}, Max stages: {max_d}")
    print(f"  RT range: [{rt_data.min():.4f}, {rt_data.max():.4f}]")
    print(f"  Upper choices: {np.sum(choice_data == 1)} "
          f"({100*np.mean(choice_data == 1):.1f}%)")

    save_simulated_data(sim_id, true_params, rt_data, choice_data, d_data,
                        r1_data, r2_data, flag_data, mu_data_padded,
                        sacc_data_padded, mu1_data, mu2_data, length_data, output_dir)

    results = {
        'sim_id': sim_id,
        'true_params': true_params,
        'angle_posterior': {},
        'addm_posterior': {},
        'timing': {
            'data_simulation': data_time,
            'angle_inference': 0.0,
            'addm_inference': 0.0,
        }
    }

    # Run Angle inference
    print(f"\nRunning Angle inference (HSSM Angle likelihood, TADA drift)...")
    angle_start = time.time()
    try:
        angle_model = build_angle_model(rt_data, choice_data, r1_data, r2_data,
                                        flag_data, sacc_data_padded, length_data,
                                        max_d, true_params)
        angle_results = run_angle_inference(angle_model, seed * 10000 + sim_id)
        results['angle_posterior'] = angle_results
        results['timing']['angle_inference'] = time.time() - angle_start
        print(f"Angle inference completed in {results['timing']['angle_inference']:.1f}s")
        print(f"Angle Posterior Means: {angle_results}")
    except Exception as e:
        print(f"Angle inference failed: {e}")
        results['angle_posterior'] = {}

    # Run aDDM inference
    if JAX_AVAILABLE:
        print(f"\nRunning aDDM inference (JAX likelihood)...")
        addm_start = time.time()
        try:
            addm_model, _ = build_addm_jax_model(rt_data, choice_data, r1_data, r2_data,
                                                 flag_data, sacc_data_padded, length_data,
                                                 max_d, true_params)
            addm_results = run_addm_inference(addm_model, seed * 20000 + sim_id)
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
    sim_params_dir = output_dir / "simulated_params" / f"sim_{sim_id:05d}"
    sim_params_dir.mkdir(parents=True, exist_ok=True)

    true_params_serializable = {k: float(v) for k, v in true_params.items()}
    with open(sim_params_dir / "true_params.json", 'w') as f:
        json.dump(true_params_serializable, f, indent=2)

    np.savez(
        sim_params_dir / "simulated_data.npz",
        rt_data=rt_data,
        choice_data=choice_data,
        d_data=d_data,
        r1_data=r1_data,
        r2_data=r2_data,
        flag_data=flag_data,
        mu_data_padded=mu_data_padded,
        sacc_data_padded=sacc_data_padded,
        mu1_data=mu1_data,
        mu2_data=mu2_data,
        length_data=length_data,
    )
    print(f"Simulated parameters saved to {sim_params_dir}/")


def save_results(results: Dict[str, Any], output_dir: Path) -> None:
    """Save results to a structured format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_id = results['sim_id']
    json_path = output_dir / f"sim_{sim_id:05d}.json"

    results_serializable = {
        'sim_id': results['sim_id'],
        'true_params': {k: float(v) for k, v in results['true_params'].items()},
        'angle_posterior': {k: float(v) for k, v in results['angle_posterior'].items()},
        'addm_posterior': {k: float(v) for k, v in results['addm_posterior'].items()},
        'timing': results['timing'],
    }

    with open(json_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"Results saved to {json_path}")


def main():
    parser = argparse.ArgumentParser(description="aDDM vs Angle slope-recovery simulation")
    parser.add_argument('--sim-id', type=int, default=0, help='Simulation ID')
    parser.add_argument('--output-dir', type=str, default='./recovery_results_slope/',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')

    parser.add_argument('--simulate-eta', type=str, default='FALSE',
                        help='Sample eta from prior (TRUE/FALSE)')
    parser.add_argument('--eta-const', type=float, default=0.3)

    parser.add_argument('--simulate-kappa', type=str, default='FALSE',
                        help='Sample kappa from prior (TRUE/FALSE)')
    parser.add_argument('--kappa-const', type=float, default=0.5)

    parser.add_argument('--simulate-a', type=str, default='FALSE',
                        help='Sample a from prior (TRUE/FALSE)')
    parser.add_argument('--a-const', type=float, default=2.0)

    parser.add_argument('--simulate-b', type=str, default='TRUE',
                        help='Sample b (slope) from prior (TRUE/FALSE)')
    parser.add_argument('--b-const', type=float, default=0.3,
                        help='Constant b value when not sampling')

    parser.add_argument('--simulate-x0', type=str, default='FALSE',
                        help='Sample x0 from prior (TRUE/FALSE)')
    parser.add_argument('--x0-const', type=float, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    seed = args.seed + args.sim_id

    simulate_eta = args.simulate_eta.upper() == 'TRUE'
    simulate_kappa = args.simulate_kappa.upper() == 'TRUE'
    simulate_a = args.simulate_a.upper() == 'TRUE'
    simulate_b = args.simulate_b.upper() == 'TRUE'
    simulate_x0 = args.simulate_x0.upper() == 'TRUE'

    print(f"\nParameter Sampling Configuration:")
    print(f"  eta:   {'sample from prior' if simulate_eta   else f'constant ({args.eta_const})'}")
    print(f"  kappa: {'sample from prior' if simulate_kappa else f'constant ({args.kappa_const})'}")
    print(f"  a:     {'sample from prior' if simulate_a     else f'constant ({args.a_const})'}")
    print(f"  b:     {'sample from prior' if simulate_b     else f'constant ({args.b_const})'}")
    print(f"  x0:    {'sample from prior' if simulate_x0    else f'constant ({args.x0_const})'}")

    results = run_single_simulation(
        args.sim_id, output_dir, seed,
        simulate_eta, args.eta_const,
        simulate_kappa, args.kappa_const,
        simulate_a, args.a_const,
        simulate_b, args.b_const,
        simulate_x0, args.x0_const,
    )

    save_results(results, output_dir)

    print(f"\n{'='*70}")
    print(f"Simulation {args.sim_id} completed successfully!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
