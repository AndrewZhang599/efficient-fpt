#!/usr/bin/env python3
"""
pymc_sampler_comparison.py

Script version of the notebook `pymc_sampler_comparison.ipynb`.

Purpose
-------
Run PyMC sampling using either:
  1) A Cython-backed likelihood (gradient-free; Metropolis), or
  2) A JAX-backed likelihood with gradients (NUTS via numpyro backend)

Outputs
-------
- InferenceData saved to NetCDF (.nc) so you can generate diagnostics later
- Timing JSON with sampling wall time and per-draw time
- Optional diagnostic plots (PNG)

Typical usage
-------------
CPU (Cython + Metropolis):
  python pymc_sampler_comparison.py --mode cython --data-path example4_new/addm_data_20251015-163921.pkl

GPU (JAX + NUTS):
  python pymc_sampler_comparison.py --mode jax --data-path example4_new/addm_data_20251015-163921.pkl

Notes
-----
- This script assumes the same imports as the notebook:
    * efficient_fpt.multi_stage_cy (Cython)
    * efficient_fpt_jax.multi_stage (JAX)
- If these modules live in a repo and are not installed, use --src-path (or set PYTHONPATH).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

import pymc as pm
import pytensor
import pytensor.tensor as pt
import arviz as az

from pytensor.graph.op import Op

# JAX imports
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

# For plotting (optional)
import matplotlib
matplotlib.use("Agg")  # safe for headless HPC nodes
import matplotlib.pyplot as plt


# ============================================================
# Likelihood backends (Cython)
# ============================================================

def _import_cython_backend():
    # Import Cython implementation
    from efficient_fpt.multi_stage_cy import compute_loss_parallel, print_num_threads
    return compute_loss_parallel, print_num_threads


class LogLikeCython(Op):
    """
    PyTensor Op wrapping the Cython likelihood function.

    Input: theta = [eta, kappa, a, b, x0]
    Output: scalar log-likelihood

    NOTE: No grad() method defined => gradient-free only (Metropolis).
    """
    itypes = [pt.dvector]
    otypes = [pt.dscalar]

    def __init__(
        self,
        compute_loss_parallel_fn,
        rt_data,
        choice_data,
        r1_data,
        r2_data,
        flag_data,
        sacc_data,
        length_data,
        max_d,
        sigma,
        num_threads: int = 8,
    ):
        self.compute_loss_parallel_fn = compute_loss_parallel_fn
        self.rt_data = np.asarray(rt_data, dtype=np.float64)
        self.choice_data = np.asarray(choice_data, dtype=np.int32)
        self.r1_data = np.asarray(r1_data, dtype=np.float64)
        self.r2_data = np.asarray(r2_data, dtype=np.float64)
        self.flag_data = np.asarray(flag_data, dtype=np.int32)
        self.sacc_data = np.asarray(sacc_data, dtype=np.float64)
        self.length_data = np.asarray(length_data, dtype=np.int32)
        self.max_d = int(max_d)
        self.sigma = float(sigma)
        self.num_data = len(self.rt_data)
        self.num_threads = int(num_threads)

    def perform(self, node, inputs, outputs):
        (theta,) = inputs
        eta, kappa, a, b, x0 = theta

        # Compute drift rates from eta, kappa
        mu1_data = kappa * (self.r1_data - eta * self.r2_data)
        mu2_data = kappa * (eta * self.r1_data - self.r2_data)

        # Compute negative log-likelihood (implementation-specific)
        nll = self.compute_loss_parallel_fn(
            mu1_data, mu2_data,
            self.rt_data, self.choice_data, self.flag_data,
            self.sacc_data, self.length_data, self.max_d,
            self.sigma, a, b, x0,
            num_threads=self.num_threads,
        )

        # Return total log-likelihood (negative of NLL)
        loglik = -self.num_data * nll
        outputs[0][0] = np.array(loglik, dtype="float64")


# ============================================================
# Likelihood backends (JAX) + gradients for NUTS
# ============================================================

def _import_jax_backend():
    # Import JAX implementation
    from efficient_fpt_jax.multi_stage import get_addm_fptd_jax_fast, pad_sacc_array_safely
    return get_addm_fptd_jax_fast, pad_sacc_array_safely


from pytensor.link.jax.dispatch import jax_funcify


class LogLikeJAX(Op):
    """
    PyTensor Op wrapping a JAX log-likelihood function WITH GRADIENTS.

    This enables NUTS sampling by providing gradient information.
    """
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
    """Gradient Op for the JAX likelihood."""
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


# Register JAX implementations for our custom Ops so PyMC's JAX-backed samplers can use them.
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


# ============================================================
# Model + I/O helpers
# ============================================================

def add_src_to_path(src_path: str | None) -> None:
    if not src_path:
        return
    src = Path(src_path).expanduser().resolve()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def load_and_prepare_data(
    data_path: str,
    num_trials: int,
    random_seed: int,
) -> Dict[str, Any]:
    data = pickle.load(open(data_path, "rb"))

    true_params = {
        "eta": float(data["eta"]),
        "kappa": float(data["kappa"]),
        "a": float(data["a"]),
        "b": float(data["b"]),
        "x0": float(data["x0"]),
        "sigma": float(data["sigma"]),
    }

    r1_full = data["r1_data"]
    r2_full = data["r2_data"]
    flag_full = data["flag_data"].astype(np.int32)
    sacc_full = data["sacc_array_padded_data"].astype(np.float64)
    length_full = data["d_data"].astype(np.int32)
    rt_full = data["decision_data"][:, 0].astype(np.float64)
    choice_full = data["decision_data"][:, 1].astype(np.int32)

    num_data_full, max_d = sacc_full.shape

    # Subset to num_trials
    rng = np.random.default_rng(random_seed)
    num_trials = min(int(num_trials), int(num_data_full))

    if num_trials >= num_data_full: 
        idx = np.arange(num_data_full)
    else: 
        idx = rng.choice(num_data_full, size=num_trials, replace=False)

    r1_data = r1_full[idx]
    r2_data = r2_full[idx]
    flag_data = flag_full[idx]
    sacc_data = sacc_full[idx]
    length_data = length_full[idx]
    rt_data = rt_full[idx]
    choice_data = choice_full[idx]

    num_data = len(rt_data)
    sigma = true_params["sigma"]
    M = float(np.max(rt_data))  # Max RT for constraints

    return {
        "TRUE_PARAMS": true_params,
        "r1_data": r1_data,
        "r2_data": r2_data,
        "flag_data": flag_data,
        "sacc_data": sacc_data,
        "length_data": length_data,
        "rt_data": rt_data,
        "choice_data": choice_data,
        "num_data": num_data,
        "max_d": int(max_d),
        "sigma": float(sigma),
        "M": float(M),
    }


def build_pymc_model(loglike_op, M: float, name_suffix: str = "") -> pm.Model:
    """
    Build a PyMC model with the given likelihood Op.
    Priors are weakly informative but proper.
    """
    with pm.Model() as model:
        # eta: attention discount factor in [0, 1]
        eta = pm.Beta(f"eta{name_suffix}", alpha=2.0, beta=2.0)

        # kappa: drift scaling (positive)
        kappa = pm.Gamma(f"kappa{name_suffix}", alpha=2.0, beta=4.0)

        # a: initial boundary (positive)
        a = pm.Gamma(f"a{name_suffix}", alpha=4.0, beta=2.0)

        # b: boundary collapse rate; constrain so boundary doesn't collapse before max RT:
        #   a - b*M > 0  =>  b < a/M
        b_raw = pm.Beta(f"b_raw{name_suffix}", alpha=2.0, beta=2.0)
        b = pm.Deterministic(f"b{name_suffix}", b_raw * a / M * 0.95)

        # x0: starting position (between -a and a)
        x0_raw = pm.Beta(f"x0_raw{name_suffix}", alpha=2.0, beta=2.0)
        x0 = pm.Deterministic(f"x0{name_suffix}", -a + 2.0 * a * x0_raw)

        theta = pt.stack([eta, kappa, a, b, x0])

        pm.Potential(f"loglik{name_suffix}", loglike_op(theta))

    return model


def ensure_outdir(base_outdir: str | None, mode: str, tag: str | None) -> Path:
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"sampler_{mode}_{ts}" + (f"_{tag}" if tag else "")
    outdir = Path(base_outdir).expanduser().resolve() if base_outdir else Path.cwd() / "results" / name
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def save_json(obj: Dict[str, Any], path: Path) -> None:
    def _default(x):
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        return str(x)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default))


def save_diagnostic_plots(idata: az.InferenceData, outdir: Path, label: str) -> None:
    params = ["eta", "kappa", "a", "b", "x0"]

    # Trace
    ax = az.plot_trace(idata, var_names=params)
    plt.tight_layout()
    plt.savefig(outdir / f"trace_{label}.png", dpi=200)
    plt.close()

    # Rank plot (robust convergence view)
    ax = az.plot_rank(idata, var_names=params)
    plt.tight_layout()
    plt.savefig(outdir / f"rank_{label}.png", dpi=200)
    plt.close()

    # Autocorr
    ax = az.plot_autocorr(idata, var_names=params)
    plt.tight_layout()
    plt.savefig(outdir / f"autocorr_{label}.png", dpi=200)
    plt.close()

    # Pair plot can be heavy; keep it optional via caller (but still useful if chains are small)
    try:
        ax = az.plot_pair(idata, var_names=params, kind="kde", marginals=True)
        plt.tight_layout()
        plt.savefig(outdir / f"pair_{label}.png", dpi=200)
        plt.close()
    except Exception as e:
        (outdir / f"pair_{label}.txt").write_text(f"Pair plot failed: {type(e).__name__}: {e}\n")

    # Energy plot only exists for NUTS-like samplers
    try:
        ax = az.plot_energy(idata)
        plt.tight_layout()
        plt.savefig(outdir / f"energy_{label}.png", dpi=200)
        plt.close()
    except Exception:
        pass


# ============================================================
# Backend builders
# ============================================================

def build_cython_loglike_op(data: Dict[str, Any], num_threads: int) -> Op:
    compute_loss_parallel, print_num_threads = _import_cython_backend()
    print("Cython backend loaded.")
    try:
        print_num_threads()
    except Exception:
        pass

    return LogLikeCython(
        compute_loss_parallel_fn=compute_loss_parallel,
        rt_data=data["rt_data"],
        choice_data=data["choice_data"],
        r1_data=data["r1_data"],
        r2_data=data["r2_data"],
        flag_data=data["flag_data"],
        sacc_data=data["sacc_data"],
        length_data=data["length_data"],
        max_d=data["max_d"],
        sigma=data["sigma"],
        num_threads=num_threads,
    )


def build_jax_loglike_op(
    data: Dict[str, Any],
    trunc_num: int,
    warmup: bool = True,
) -> Tuple[Op, Dict[str, float]]:
    """
    Returns (jax_loglike_op, timing_dict).
    timing_dict includes warmup_compile_seconds (if warmup=True).
    """
    get_addm_fptd_jax_fast, pad_sacc_array_safely = _import_jax_backend()
    print("JAX backend loaded.")

    # Convert data to JAX arrays
    jax_rt = jnp.array(data["rt_data"])
    jax_choice = jnp.array(data["choice_data"])
    jax_d = jnp.array(data["length_data"])
    jax_r1 = jnp.array(data["r1_data"])
    jax_r2 = jnp.array(data["r2_data"])
    jax_flag = jnp.array(data["flag_data"])
    jax_sacc = jnp.array(data["sacc_data"])

    max_d = int(data["max_d"])
    sigma = float(data["sigma"])
    trunc_num = int(trunc_num)

    # Pre-compute safe saccade arrays (avoids NaN in gradients)
    print("Pre-computing safe saccade arrays for JAX...")
    jax_sacc_safe = vmap(lambda s, d: pad_sacc_array_safely(s, d, max_d))(jax_sacc, jax_d)
    jax_sacc_safe.block_until_ready()
    print("Done.")

    def compute_mu_arrays_jax(eta, kappa, r1, r2, flag, max_d_local):
        mu1 = kappa * (r1 - eta * r2)
        mu2 = kappa * (eta * r1 - r2)
        indices = jnp.arange(max_d_local)
        mu_array = jnp.where((indices % 2) == flag, mu1, mu2)
        return mu_array

    def jax_loglik_single(rt, choice, d, r1, r2, flag, sacc_safe, eta, kappa, a, b, x0):
        mu_array = compute_mu_arrays_jax(eta, kappa, r1, r2, flag, max_d)
        fptd = get_addm_fptd_jax_fast(
            rt, d, mu_array, sacc_safe, sigma, a, b, x0, choice,
            order=30, trunc_num=trunc_num, safe_sacc=sacc_safe
        )
        return jnp.log(jnp.maximum(fptd, 1e-30))

    def jax_loglik_batch(eta, kappa, a, b, x0):
        loglik_fn = vmap(
            lambda rt, choice, d, r1, r2, flag, sacc_safe: jax_loglik_single(
                rt, choice, d, r1, r2, flag, sacc_safe, eta, kappa, a, b, x0
            )
        )
        logliks = loglik_fn(jax_rt, jax_choice, jax_d, jax_r1, jax_r2, jax_flag, jax_sacc_safe)
        return jnp.sum(logliks)

    jax_loglik_jit = jit(jax_loglik_batch)
    jax_grad_loglik = jit(grad(jax_loglik_batch, argnums=(0, 1, 2, 3, 4)))

    timing = {"warmup_compile_seconds": 0.0}

    if warmup:
        tp = data["TRUE_PARAMS"]
        print("Warming up JAX JIT compilation...")
        t0 = time.perf_counter()
        _ = jax_loglik_jit(tp["eta"], tp["kappa"], tp["a"], tp["b"], tp["x0"])
        _ = jax_grad_loglik(tp["eta"], tp["kappa"], tp["a"], tp["b"], tp["x0"])
        # Ensure compilation completes
        jax.block_until_ready(_)
        timing["warmup_compile_seconds"] = time.perf_counter() - t0
        print("Done.")

    jax_loglike_op = LogLikeJAX(jax_loglik_jit, jax_grad_loglik)
    return jax_loglike_op, timing


# ============================================================
# Runners
# ============================================================

def run_cython_metropolis(
    model: pm.Model,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    random_seed: int,
    mp_ctx: str | None,
) -> Tuple[az.InferenceData, Dict[str, float]]:
    t0 = time.perf_counter()
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            step=pm.Metropolis(),
            random_seed=random_seed,
            progressbar=True,
            return_inferencedata=True,
            mp_ctx=mp_ctx,
        )
    elapsed = time.perf_counter() - t0
    return idata, {"sampling_seconds": elapsed, "time_per_draw_ms": 1000.0 * elapsed / (draws * chains)}


def run_jax_nuts_numpyro(
    model: pm.Model,
    draws: int,
    tune: int,
    chains: int,
    random_seed: int,
) -> Tuple[az.InferenceData, Dict[str, float]]:
    t0 = time.perf_counter()
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            nuts_sampler="numpyro",
            random_seed=random_seed,
            progressbar=True,
            return_inferencedata=True,
        )
    elapsed = time.perf_counter() - t0
    return idata, {"sampling_seconds": elapsed, "time_per_draw_ms": 1000.0 * elapsed / (draws * chains)}


def rename_vars_for_standard_plots(idata: az.InferenceData, suffix: str) -> az.InferenceData:
    # The notebook used suffixes _cy and _jax; for a single-mode run, normalize names.
    rename_map = {
        f"eta{suffix}": "eta",
        f"kappa{suffix}": "kappa",
        f"a{suffix}": "a",
        f"b{suffix}": "b",
        f"x0{suffix}": "x0",
    }
    try:
        return idata.rename(rename_map)
    except Exception:
        return idata


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["cython", "jax", "both"], default="cython",
                   help="Which backend to run.")
    p.add_argument("--data-path", required=True, help="Path to the pickled dataset (.pkl).")
    p.add_argument("--num-trials", type=int, default=500, help="Number of trials to subsample.")
    p.add_argument("--draws", type=int, default=500, help="Posterior draws per chain.")
    p.add_argument("--tune", type=int, default=200, help="Tuning steps.")
    p.add_argument("--chains", type=int, default=2, help="Number of chains.")
    p.add_argument("--random-seed", type=int, default=42, help="RNG seed for reproducibility.")

    p.add_argument("--num-threads", type=int, default=8, help="OpenMP threads for Cython backend.")
    p.add_argument("--trunc-num", type=int, default=6, help="Series truncation for JAX backend.")
    p.add_argument("--cores", type=int, default=2,
                   help="PyMC 'cores' argument for Metropolis run (Cython uses OpenMP internally).")
    p.add_argument("--mp-ctx", default="spawn", help="PyMC multiprocessing context (e.g., spawn/forkserver).")

    p.add_argument("--src-path", default=None,
                   help="Optional path to add to PYTHONPATH (e.g., ../src) if modules are not installed.")
    p.add_argument("--outdir", default=None,
                   help="Output directory; if omitted, creates results/sampler_<mode>_<timestamp>/")
    p.add_argument("--tag", default=None, help="Optional string appended to output directory name.")
    p.add_argument("--save-plots", action="store_true", help="Save diagnostic plots as PNGs.")
    p.add_argument("--no-jax-warmup", action="store_true", help="Skip explicit JAX warmup compilation.")

    # Optional environment knobs (left off by default; set if you want)
    p.add_argument("--jax-debug-nans", action="store_true", help="Enable jax_debug_nans.")
    p.add_argument("--jax-debug-infs", action="store_true", help="Enable jax_debug_infs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Optional: add src to PYTHONPATH
    add_src_to_path(args.src_path)

    if args.jax_debug_nans:
        jax.config.update("jax_debug_nans", True)
    if args.jax_debug_infs:
        jax.config.update("jax_debug_infs", True)

    print(f"PyMC version: {pm.__version__}")
    print(f"JAX version: {jax.__version__}")
    try:
        print(f"JAX devices: {jax.devices()}")
    except Exception:
        pass

    data = load_and_prepare_data(args.data_path, args.num_trials, args.random_seed)
    print(f"Data loaded: {data['num_data']} trials, max_d={data['max_d']}")
    print(f"Max RT (M): {data['M']:.6f}")
    print("True parameters:")
    for k, v in data["TRUE_PARAMS"].items():
        print(f"  {k}: {v}")

    outdir = ensure_outdir(args.outdir, args.mode, args.tag)
    print(f"Output directory: {outdir}")

    # Persist config + environment snapshot
    cfg = {
        "mode": args.mode,
        "data_path": str(Path(args.data_path).expanduser().resolve()),
        "num_trials": args.num_trials,
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "random_seed": args.random_seed,
        "num_threads": args.num_threads,
        "trunc_num": args.trunc_num,
        "cores": args.cores,
        "mp_ctx": args.mp_ctx,
        "pymc_version": pm.__version__,
        "jax_version": jax.__version__,
        "timestamp": _dt.datetime.now().isoformat(),
        "hostname": os.uname().nodename if hasattr(os, "uname") else None,
        "env": {
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "XLA_PYTHON_CLIENT_ALLOCATOR": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR"),
            "JAX_PLATFORM_NAME": os.environ.get("JAX_PLATFORM_NAME"),
        },
    }
    save_json(cfg, outdir / "config.json")

    results: Dict[str, Any] = {}

    if args.mode in ("cython", "both"):
        print("=" * 60)
        print("Cython + Metropolis Sampling")
        print("=" * 60)
        cython_op = build_cython_loglike_op(data, num_threads=args.num_threads)
        model_cy = build_pymc_model(cython_op, M=data["M"], name_suffix="_cy")

        # Sanity check loglik at true params
        theta_true = np.array([data["TRUE_PARAMS"][k] for k in ["eta", "kappa", "a", "b", "x0"]], dtype=float)
        theta_sym = pt.dvector("theta")
        ll_cy_fn = pytensor.function([theta_sym], cython_op(theta_sym))
        print(f"Cython log-likelihood at true params: {ll_cy_fn(theta_true):.6f}")

        idata_cy, timing_cy = run_cython_metropolis(
            model=model_cy,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            random_seed=args.random_seed,
            mp_ctx=args.mp_ctx if args.mp_ctx else None,
        )
        idata_cy = rename_vars_for_standard_plots(idata_cy, "_cy")
        idata_path = outdir / "idata_cython.nc"
        idata_cy.to_netcdf(idata_path)
        save_json(timing_cy, outdir / "timing_cython.json")
        results["cython"] = {"idata": str(idata_path), **timing_cy}
        print(f"Cython sampling seconds: {timing_cy['sampling_seconds']:.3f}")
        print(f"Cython time per draw (ms): {timing_cy['time_per_draw_ms']:.3f}")

        if args.save_plots:
            save_diagnostic_plots(idata_cy, outdir, "cython")

    if args.mode in ("jax", "both"):
        print("=" * 60)
        print("JAX + NUTS Sampling (numpyro backend)")
        print("=" * 60)
        jax_op, jax_build_timing = build_jax_loglike_op(
            data,
            trunc_num=args.trunc_num,
            warmup=(not args.no_jax_warmup),
        )
        model_jx = build_pymc_model(jax_op, M=data["M"], name_suffix="_jax")

        # Sanity checks
        theta_true = np.array([data["TRUE_PARAMS"][k] for k in ["eta", "kappa", "a", "b", "x0"]], dtype=float)
        theta_sym = pt.dvector("theta")
        ll_jx_fn = pytensor.function([theta_sym], jax_op(theta_sym))
        print(f"JAX Op log-likelihood at true params: {ll_jx_fn(theta_true):.6f}")
        grad_sym = pt.grad(jax_op(theta_sym), theta_sym)
        grad_fn = pytensor.function([theta_sym], grad_sym)
        print(f"JAX Op gradients at true params: {grad_fn(theta_true)}")

        idata_jx, timing_jx = run_jax_nuts_numpyro(
            model=model_jx,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            random_seed=args.random_seed,
        )
        idata_jx = rename_vars_for_standard_plots(idata_jx, "_jax")
        idata_path = outdir / "idata_jax.nc"
        idata_jx.to_netcdf(idata_path)

        timing_full = {**jax_build_timing, **timing_jx}
        save_json(timing_full, outdir / "timing_jax.json")
        results["jax"] = {"idata": str(idata_path), **timing_full}

        print(f"JAX warmup compile seconds: {jax_build_timing.get('warmup_compile_seconds', 0.0):.3f}")
        print(f"JAX sampling seconds: {timing_jx['sampling_seconds']:.3f}")
        print(f"JAX time per draw (ms): {timing_jx['time_per_draw_ms']:.3f}")

        if args.save_plots:
            save_diagnostic_plots(idata_jx, outdir, "jax")

    save_json(results, outdir / "results_summary.json")
    print("=" * 60)
    print("Done.")
    print(f"Results summary saved to: {outdir / 'results_summary.json'}")


if __name__ == "__main__":
    main()
