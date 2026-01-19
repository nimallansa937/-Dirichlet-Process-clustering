#!/usr/bin/env python3
"""
Pitman-Yor Process vs Dirichlet Process Clustering Validation
==============================================================

This script validates claims that PYP outperforms DP on power-law distributed data:
- PYP beats DP by 17-22% on power-law data (NMI metric)
- DP underestimates cluster count on power-law distributions
- DP "collapses" niche clusters into large ones
- PYP sustains discovery as n grows

Run on NVIDIA B200 GPU for best performance at scale.

Usage:
    python validate_pyp_vs_dp.py --n_samples 50000 --n_features 60 --n_clusters 30
    python validate_pyp_vs_dp.py --run_sweep  # Full parameter sweep

Author: HIMARI Project
"""

import argparse
import time
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import zipf as zipf_dist
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score
)
from sklearn.mixture import BayesianGaussianMixture

# Suppress sklearn convergence warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Check for GPU availability
try:
    import torch
    import pyro
    import pyro.distributions as dist
    from pyro.infer import SVI, TraceEnum_ELBO, Trace_ELBO
    from pyro.optim import Adam, ClippedAdam
    PYRO_AVAILABLE = True
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        DEVICE = torch.device('cuda')
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        DEVICE = torch.device('cpu')
        GPU_NAME = "CPU"
        GPU_MEMORY = 0
except ImportError:
    PYRO_AVAILABLE = False
    GPU_AVAILABLE = False
    DEVICE = None
    GPU_NAME = "N/A"
    GPU_MEMORY = 0


# ═══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SyntheticDataset:
    """Container for synthetic clustering dataset."""
    X: np.ndarray
    y_true: np.ndarray
    cluster_sizes: np.ndarray
    n_true_clusters: int
    zipf_exponent: float

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def size_distribution_summary(self) -> str:
        """Return summary of cluster size distribution."""
        sorted_sizes = sorted(self.cluster_sizes, reverse=True)
        top_5 = sorted_sizes[:5]
        bottom_5 = sorted_sizes[-5:]
        return f"Top 5: {top_5}, Bottom 5: {bottom_5}"


def generate_power_law_clusters(
    n_samples: int = 50_000,
    n_features: int = 60,
    zipf_exponent: float = 1.5,
    n_true_clusters: int = 30,
    cluster_separation: float = 3.0,
    intra_cluster_std: float = 0.5,
    seed: int = 42
) -> SyntheticDataset:
    """
    Generate synthetic data with power-law distributed cluster sizes.

    This mimics the structure of trading strategy populations where:
    - A few dominant strategies have many variations
    - Many niche strategies have few variations
    - Distribution follows Zipf's law

    Parameters
    ----------
    n_samples : int
        Total number of samples to generate
    n_features : int
        Dimensionality of feature space (60 = typical for HIMARI)
    zipf_exponent : float
        Controls steepness of power-law (1.2=flat, 2.5=steep)
    n_true_clusters : int
        Number of ground-truth clusters
    cluster_separation : float
        Distance between cluster centers (higher = easier separation)
    intra_cluster_std : float
        Standard deviation within clusters
    seed : int
        Random seed for reproducibility

    Returns
    -------
    SyntheticDataset
        Container with X, y_true, cluster_sizes, and metadata
    """
    np.random.seed(seed)

    # Generate cluster sizes following Zipf's law
    # Zipf: P(k) ∝ 1/k^s where s = zipf_exponent
    raw_sizes = zipf_dist.rvs(zipf_exponent, size=n_true_clusters, random_state=seed)

    # Normalize to sum to n_samples
    cluster_sizes = (raw_sizes / raw_sizes.sum() * n_samples).astype(int)

    # Ensure we have exactly n_samples (adjust largest cluster)
    diff = n_samples - cluster_sizes.sum()
    cluster_sizes[0] += diff

    # Ensure no zero-size clusters
    cluster_sizes = np.maximum(cluster_sizes, 1)
    actual_n = cluster_sizes.sum()

    # Generate well-separated cluster centers in high-dimensional space
    # Using orthogonal-ish directions for better separation
    centers = np.random.randn(n_true_clusters, n_features) * cluster_separation

    # Generate samples for each cluster
    X_list = []
    labels_list = []

    for k, size in enumerate(cluster_sizes):
        # Gaussian blob around cluster center
        samples = centers[k] + np.random.randn(size, n_features) * intra_cluster_std
        X_list.append(samples)
        labels_list.extend([k] * size)

    X = np.vstack(X_list)
    labels = np.array(labels_list)

    # Shuffle to remove ordering bias
    perm = np.random.permutation(actual_n)
    X = X[perm]
    labels = labels[perm]

    return SyntheticDataset(
        X=X,
        y_true=labels,
        cluster_sizes=cluster_sizes,
        n_true_clusters=n_true_clusters,
        zipf_exponent=zipf_exponent
    )


# ═══════════════════════════════════════════════════════════════════════════════
# METHOD 1: STANDARD DIRICHLET PROCESS (scikit-learn)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClusteringResult:
    """Container for clustering results."""
    labels: np.ndarray
    k_inferred: int
    elapsed_time: float
    method: str
    extra_info: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra_info is None:
            self.extra_info = {}


def fit_dirichlet_process(
    X: np.ndarray,
    max_components: int = 100,
    concentration: float = 1.0,
    n_init: int = 3,
    max_iter: int = 500,
    random_state: int = 42
) -> ClusteringResult:
    """
    Fit standard Dirichlet Process Mixture Model using variational inference.

    Uses scikit-learn's BayesianGaussianMixture with DP prior.
    This is the baseline we're comparing against.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_samples, n_features)
    max_components : int
        Maximum number of mixture components (truncation level)
    concentration : float
        DP concentration parameter α (higher = more clusters expected)
    n_init : int
        Number of random initializations
    max_iter : int
        Maximum EM iterations
    random_state : int
        Random seed

    Returns
    -------
    ClusteringResult
        Labels, inferred K, timing, and metadata
    """
    start = time.time()

    model = BayesianGaussianMixture(
        n_components=max_components,
        covariance_type='full',
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=concentration,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        verbose=0
    )

    model.fit(X)
    elapsed = time.time() - start

    labels = model.predict(X)
    k_inferred = len(np.unique(labels))

    # Get weight distribution for analysis
    weights = model.weights_
    active_weights = weights[weights > 0.01]  # Clusters with >1% weight

    return ClusteringResult(
        labels=labels,
        k_inferred=k_inferred,
        elapsed_time=elapsed,
        method="DP (sklearn)",
        extra_info={
            'n_active_components': len(active_weights),
            'weight_concentration': model.weight_concentration_,
            'converged': model.converged_,
            'n_iter': model.n_iter_
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# METHOD 2: PITMAN-YOR PROCESS (Pyro on GPU)
# ═══════════════════════════════════════════════════════════════════════════════

def fit_pitman_yor_gpu(
    X: np.ndarray,
    max_components: int = 100,
    discount: float = 0.25,
    concentration: float = 1.0,
    n_steps: int = 1000,
    learning_rate: float = 0.01,
    batch_size: Optional[int] = None,
    verbose: bool = True
) -> ClusteringResult:
    """
    Fit Pitman-Yor Process Mixture Model using variational inference on GPU.

    Key difference from DP:
    - DP:  β_k ~ Beta(1, α)
    - PYP: β_k ~ Beta(1 - d, α + k*d)  where d = discount

    The discount parameter d ∈ [0, 1) creates a "rich-get-richer" effect
    that is modulated, allowing for better modeling of power-law tails.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_samples, n_features)
    max_components : int
        Truncation level for stick-breaking
    discount : float
        PYP discount parameter d ∈ [0, 1). d=0 reduces to DP.
    concentration : float
        Base concentration parameter α
    n_steps : int
        Number of SVI optimization steps
    learning_rate : float
        Adam optimizer learning rate
    batch_size : int, optional
        Mini-batch size. None = full batch.
    verbose : bool
        Print progress

    Returns
    -------
    ClusteringResult
        Labels, inferred K, timing, and metadata
    """
    if not PYRO_AVAILABLE:
        raise ImportError("Pyro not available. Install with: pip install pyro-ppl")

    # Clear any previous state
    pyro.clear_param_store()

    # Move data to GPU
    X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    N, D = X_tensor.shape
    K = max_components

    start = time.time()

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL: Pitman-Yor Process with stick-breaking construction
    # ─────────────────────────────────────────────────────────────────────────

    def model(X_batch):
        batch_size = X_batch.shape[0]

        # Pitman-Yor stick-breaking prior
        # β_k ~ Beta(1 - d, α + k*d) for k = 1, ..., K-1
        with pyro.plate('sticks', K - 1):
            k_idx = torch.arange(K - 1, device=DEVICE, dtype=torch.float32)
            beta_a = torch.ones(K - 1, device=DEVICE) * (1.0 - discount)
            beta_b = concentration + (k_idx + 1) * discount
            v = pyro.sample('v', dist.Beta(beta_a, beta_b))

        # Convert stick lengths to mixture weights via stick-breaking
        # π_k = v_k * ∏_{j<k}(1 - v_j)
        one_minus_v = 1.0 - v
        cumprod = torch.cumprod(one_minus_v, dim=0)
        cumprod_shifted = torch.cat([torch.ones(1, device=DEVICE), cumprod[:-1]])

        weights = torch.zeros(K, device=DEVICE)
        weights[:K-1] = v * cumprod_shifted
        weights[K-1] = cumprod[-1]  # Remaining mass

        # Cluster location priors (vague)
        with pyro.plate('components', K):
            locs = pyro.sample(
                'locs',
                dist.Normal(
                    torch.zeros(D, device=DEVICE),
                    torch.ones(D, device=DEVICE) * 5.0
                ).to_event(1)
            )
            # Log-normal prior on scales for positivity
            scales = pyro.sample(
                'scales',
                dist.LogNormal(
                    torch.zeros(D, device=DEVICE),
                    torch.ones(D, device=DEVICE) * 0.5
                ).to_event(1)
            )

        # Observations
        with pyro.plate('data', batch_size):
            # Mixture assignment
            assignment = pyro.sample('assignment', dist.Categorical(weights))
            # Observation likelihood
            pyro.sample(
                'obs',
                dist.Normal(locs[assignment], scales[assignment]).to_event(1),
                obs=X_batch
            )

    # ─────────────────────────────────────────────────────────────────────────
    # GUIDE: Mean-field variational approximation
    # ─────────────────────────────────────────────────────────────────────────

    def guide(X_batch):
        # Variational parameters for stick-breaking
        v_alpha = pyro.param(
            'v_alpha',
            torch.ones(K - 1, device=DEVICE) * 2.0,
            constraint=dist.constraints.positive
        )
        v_beta = pyro.param(
            'v_beta',
            torch.ones(K - 1, device=DEVICE) * concentration,
            constraint=dist.constraints.positive
        )

        with pyro.plate('sticks', K - 1):
            pyro.sample('v', dist.Beta(v_alpha, v_beta))

        # Variational parameters for cluster locations
        loc_mean = pyro.param(
            'loc_mean',
            torch.zeros(K, D, device=DEVICE)
        )
        loc_scale = pyro.param(
            'loc_scale',
            torch.ones(K, D, device=DEVICE) * 0.5,
            constraint=dist.constraints.positive
        )

        # Variational parameters for cluster scales
        scale_loc = pyro.param(
            'scale_loc',
            torch.zeros(K, D, device=DEVICE)
        )
        scale_scale = pyro.param(
            'scale_scale',
            torch.ones(K, D, device=DEVICE) * 0.3,
            constraint=dist.constraints.positive
        )

        with pyro.plate('components', K):
            pyro.sample('locs', dist.Normal(loc_mean, loc_scale).to_event(1))
            pyro.sample('scales', dist.LogNormal(scale_loc, scale_scale).to_event(1))

    # ─────────────────────────────────────────────────────────────────────────
    # SVI OPTIMIZATION
    # ─────────────────────────────────────────────────────────────────────────

    optimizer = ClippedAdam({'lr': learning_rate, 'clip_norm': 10.0})
    elbo = Trace_ELBO()
    svi = SVI(model, guide, optimizer, loss=elbo)

    losses = []

    if batch_size is None or batch_size >= N:
        # Full batch training
        for step in range(n_steps):
            loss = svi.step(X_tensor)
            losses.append(loss)

            if verbose and (step + 1) % 200 == 0:
                print(f"  Step {step + 1}/{n_steps}: ELBO loss = {loss:.2f}")
    else:
        # Mini-batch training
        n_batches = (N + batch_size - 1) // batch_size
        for step in range(n_steps):
            epoch_loss = 0.0
            perm = torch.randperm(N, device=DEVICE)

            for b in range(n_batches):
                batch_idx = perm[b * batch_size:(b + 1) * batch_size]
                X_batch = X_tensor[batch_idx]
                loss = svi.step(X_batch)
                epoch_loss += loss

            losses.append(epoch_loss / n_batches)

            if verbose and (step + 1) % 200 == 0:
                print(f"  Step {step + 1}/{n_steps}: ELBO loss = {losses[-1]:.2f}")

    elapsed = time.time() - start

    # ─────────────────────────────────────────────────────────────────────────
    # POSTERIOR INFERENCE: Assign labels
    # ─────────────────────────────────────────────────────────────────────────

    with torch.no_grad():
        # Get posterior mean weights
        v_alpha = pyro.param('v_alpha')
        v_beta = pyro.param('v_beta')
        v_mean = v_alpha / (v_alpha + v_beta)

        one_minus_v = 1.0 - v_mean
        cumprod = torch.cumprod(one_minus_v, dim=0)
        cumprod_shifted = torch.cat([torch.ones(1, device=DEVICE), cumprod[:-1]])

        weights = torch.zeros(K, device=DEVICE)
        weights[:K-1] = v_mean * cumprod_shifted
        weights[K-1] = cumprod[-1]

        # Get posterior mean locations and scales
        loc_mean = pyro.param('loc_mean')
        scale_loc = pyro.param('scale_loc')
        scales_mean = torch.exp(scale_loc)  # Mean of log-normal

        # Compute log-likelihood for each point under each component
        # Using vectorized computation for efficiency
        log_weights = torch.log(weights + 1e-10)

        # Process in chunks to avoid OOM on large datasets
        chunk_size = 10000
        labels_list = []

        for i in range(0, N, chunk_size):
            X_chunk = X_tensor[i:i + chunk_size]
            chunk_n = X_chunk.shape[0]

            # Compute squared distances to each cluster
            # Shape: (chunk_n, K, D)
            diff = X_chunk.unsqueeze(1) - loc_mean.unsqueeze(0)

            # Log-likelihood under Gaussian
            # log p(x|μ,σ) = -0.5 * D * log(2π) - D * log(σ) - 0.5 * ||x-μ||² / σ²
            log_probs = -0.5 * ((diff / scales_mean.unsqueeze(0)) ** 2).sum(dim=2)
            log_probs = log_probs - D * torch.log(scales_mean).sum(dim=1).unsqueeze(0)

            # Add log prior (weights)
            log_posterior = log_probs + log_weights.unsqueeze(0)

            # Assign to most likely cluster
            chunk_labels = log_posterior.argmax(dim=1)
            labels_list.append(chunk_labels.cpu().numpy())

        labels = np.concatenate(labels_list)

        # Count active clusters
        unique_labels = np.unique(labels)
        k_inferred = len(unique_labels)

        # Get weight distribution
        weights_np = weights.cpu().numpy()
        active_weights = weights_np[weights_np > 0.01]

    return ClusteringResult(
        labels=labels,
        k_inferred=k_inferred,
        elapsed_time=elapsed,
        method="PYP (Pyro/GPU)",
        extra_info={
            'discount': discount,
            'concentration': concentration,
            'n_steps': n_steps,
            'final_loss': losses[-1] if losses else None,
            'n_active_components': len(active_weights),
            'weight_distribution': sorted(weights_np[weights_np > 0.001], reverse=True)[:10]
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    nmi: float  # Normalized Mutual Information
    ari: float  # Adjusted Rand Index
    k_error: int  # |K_inferred - K_true|
    k_ratio: float  # K_inferred / K_true
    silhouette: Optional[float]  # Silhouette score (if computable)

    def to_dict(self) -> Dict[str, float]:
        return {
            'NMI': self.nmi,
            'ARI': self.ari,
            'K_error': self.k_error,
            'K_ratio': self.k_ratio,
            'Silhouette': self.silhouette
        }


def evaluate_clustering(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k_true: int,
    k_pred: int,
    compute_silhouette: bool = True
) -> EvaluationResult:
    """
    Compute clustering evaluation metrics.

    Parameters
    ----------
    X : np.ndarray
        Data matrix
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    k_true : int
        True number of clusters
    k_pred : int
        Inferred number of clusters
    compute_silhouette : bool
        Whether to compute silhouette (slow for large N)

    Returns
    -------
    EvaluationResult
        Container with all metrics
    """
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    k_error = abs(k_pred - k_true)
    k_ratio = k_pred / k_true

    silhouette = None
    if compute_silhouette and len(X) <= 50000:
        try:
            # Subsample for silhouette if still too large
            if len(X) > 10000:
                idx = np.random.choice(len(X), 10000, replace=False)
                silhouette = silhouette_score(X[idx], y_pred[idx])
            else:
                silhouette = silhouette_score(X, y_pred)
        except Exception:
            pass

    return EvaluationResult(
        nmi=nmi,
        ari=ari,
        k_error=k_error,
        k_ratio=k_ratio,
        silhouette=silhouette
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_comparison(
    n_samples: int = 50000,
    n_features: int = 60,
    n_true_clusters: int = 30,
    zipf_exponent: float = 1.5,
    max_components: int = 100,
    pyp_discount: float = 0.25,
    concentration: float = 1.0,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run a single head-to-head comparison of DP vs PYP.

    Returns
    -------
    dict
        Results including metrics for both methods and comparison
    """
    if verbose:
        print("=" * 70)
        print(f"EXPERIMENT: n={n_samples}, d={n_features}, K={n_true_clusters}, zipf={zipf_exponent}")
        print("=" * 70)

    # Generate data
    if verbose:
        print("\n[1/4] Generating synthetic power-law data...")
    dataset = generate_power_law_clusters(
        n_samples=n_samples,
        n_features=n_features,
        zipf_exponent=zipf_exponent,
        n_true_clusters=n_true_clusters,
        seed=seed
    )
    if verbose:
        print(f"      Cluster sizes: {dataset.size_distribution_summary()}")

    # Fit DP
    if verbose:
        print("\n[2/4] Fitting Dirichlet Process (sklearn)...")
    result_dp = fit_dirichlet_process(
        dataset.X,
        max_components=max_components,
        concentration=concentration
    )
    eval_dp = evaluate_clustering(
        dataset.X, dataset.y_true, result_dp.labels,
        n_true_clusters, result_dp.k_inferred
    )
    if verbose:
        print(f"      K_inferred={result_dp.k_inferred}, NMI={eval_dp.nmi:.4f}, "
              f"ARI={eval_dp.ari:.4f}, Time={result_dp.elapsed_time:.1f}s")

    # Fit PYP
    if PYRO_AVAILABLE:
        if verbose:
            print(f"\n[3/4] Fitting Pitman-Yor Process (Pyro on {GPU_NAME})...")
        result_pyp = fit_pitman_yor_gpu(
            dataset.X,
            max_components=max_components,
            discount=pyp_discount,
            concentration=concentration,
            n_steps=1000,
            verbose=verbose
        )
        eval_pyp = evaluate_clustering(
            dataset.X, dataset.y_true, result_pyp.labels,
            n_true_clusters, result_pyp.k_inferred
        )
        if verbose:
            print(f"      K_inferred={result_pyp.k_inferred}, NMI={eval_pyp.nmi:.4f}, "
                  f"ARI={eval_pyp.ari:.4f}, Time={result_pyp.elapsed_time:.1f}s")
    else:
        result_pyp = None
        eval_pyp = None
        if verbose:
            print("\n[3/4] Skipping PYP (Pyro not available)")

    # Summary
    if verbose:
        print("\n[4/4] Summary")
        print("-" * 50)
        print(f"True K: {n_true_clusters}")
        print(f"DP  inferred K: {result_dp.k_inferred} (error: {eval_dp.k_error})")
        if eval_pyp:
            print(f"PYP inferred K: {result_pyp.k_inferred} (error: {eval_pyp.k_error})")
            print(f"\nNMI: DP={eval_dp.nmi:.4f}, PYP={eval_pyp.nmi:.4f}")
            nmi_improvement = (eval_pyp.nmi - eval_dp.nmi) / eval_dp.nmi * 100
            print(f"NMI improvement: {nmi_improvement:+.1f}%")
            print(f"\nARI: DP={eval_dp.ari:.4f}, PYP={eval_pyp.ari:.4f}")
            ari_improvement = (eval_pyp.ari - eval_dp.ari) / (eval_dp.ari + 1e-10) * 100
            print(f"ARI improvement: {ari_improvement:+.1f}%")

    return {
        'dataset': {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_true_clusters': n_true_clusters,
            'zipf_exponent': zipf_exponent,
            'cluster_sizes': dataset.cluster_sizes.tolist()
        },
        'dp': {
            'k_inferred': result_dp.k_inferred,
            'time': result_dp.elapsed_time,
            **eval_dp.to_dict()
        },
        'pyp': {
            'k_inferred': result_pyp.k_inferred if result_pyp else None,
            'time': result_pyp.elapsed_time if result_pyp else None,
            **(eval_pyp.to_dict() if eval_pyp else {})
        } if PYRO_AVAILABLE else None
    }


def run_parameter_sweep(
    zipf_exponents: List[float] = [1.2, 1.5, 2.0, 2.5],
    sample_sizes: List[int] = [10000, 50000, 100000],
    n_features: int = 60,
    n_true_clusters: int = 30,
    seeds: List[int] = [42, 123, 456],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run parameter sweep across different conditions.

    Tests claims:
    1. PYP advantage increases with power-law steepness (higher zipf)
    2. PYP advantage holds across sample sizes
    3. DP consistently underestimates K

    Returns
    -------
    pd.DataFrame
        Results table with all experiments
    """
    results = []

    total_experiments = len(zipf_exponents) * len(sample_sizes) * len(seeds)
    exp_num = 0

    for zipf_exp in zipf_exponents:
        for n in sample_sizes:
            for seed in seeds:
                exp_num += 1
                if verbose:
                    print(f"\n{'='*70}")
                    print(f"EXPERIMENT {exp_num}/{total_experiments}")
                    print(f"zipf={zipf_exp}, n={n}, seed={seed}")
                    print('='*70)

                result = run_single_comparison(
                    n_samples=n,
                    n_features=n_features,
                    n_true_clusters=n_true_clusters,
                    zipf_exponent=zipf_exp,
                    seed=seed,
                    verbose=verbose
                )

                row = {
                    'zipf_exp': zipf_exp,
                    'n_samples': n,
                    'seed': seed,
                    'k_true': n_true_clusters,
                    'k_dp': result['dp']['k_inferred'],
                    'k_pyp': result['pyp']['k_inferred'] if result['pyp'] else None,
                    'nmi_dp': result['dp']['NMI'],
                    'nmi_pyp': result['pyp']['NMI'] if result['pyp'] else None,
                    'ari_dp': result['dp']['ARI'],
                    'ari_pyp': result['pyp']['ARI'] if result['pyp'] else None,
                    'time_dp': result['dp']['time'],
                    'time_pyp': result['pyp']['time'] if result['pyp'] else None,
                }

                # Compute improvements
                if result['pyp'] and result['pyp']['NMI']:
                    row['nmi_improvement'] = (row['nmi_pyp'] - row['nmi_dp']) / row['nmi_dp'] * 100
                    row['ari_improvement'] = (row['ari_pyp'] - row['ari_dp']) / (row['ari_dp'] + 1e-10) * 100
                    row['k_error_dp'] = abs(row['k_dp'] - n_true_clusters)
                    row['k_error_pyp'] = abs(row['k_pyp'] - n_true_clusters)

                results.append(row)

    df = pd.DataFrame(results)
    return df


def print_sweep_summary(df: pd.DataFrame):
    """Print summary statistics from parameter sweep."""
    print("\n" + "=" * 70)
    print("PARAMETER SWEEP SUMMARY")
    print("=" * 70)

    # Overall averages
    print("\n📊 OVERALL AVERAGES (across all conditions)")
    print("-" * 50)
    print(f"Mean NMI - DP:  {df['nmi_dp'].mean():.4f} ± {df['nmi_dp'].std():.4f}")
    if 'nmi_pyp' in df.columns and df['nmi_pyp'].notna().any():
        print(f"Mean NMI - PYP: {df['nmi_pyp'].mean():.4f} ± {df['nmi_pyp'].std():.4f}")
        print(f"Mean NMI improvement: {df['nmi_improvement'].mean():+.1f}%")

    print(f"\nMean K - True:  {df['k_true'].iloc[0]}")
    print(f"Mean K - DP:    {df['k_dp'].mean():.1f} ± {df['k_dp'].std():.1f}")
    if 'k_pyp' in df.columns and df['k_pyp'].notna().any():
        print(f"Mean K - PYP:   {df['k_pyp'].mean():.1f} ± {df['k_pyp'].std():.1f}")

    # By zipf exponent
    if 'nmi_improvement' in df.columns:
        print("\n📈 BY POWER-LAW STEEPNESS (zipf exponent)")
        print("-" * 50)
        by_zipf = df.groupby('zipf_exp').agg({
            'nmi_dp': 'mean',
            'nmi_pyp': 'mean',
            'nmi_improvement': 'mean',
            'k_dp': 'mean',
            'k_pyp': 'mean'
        }).round(4)
        print(by_zipf.to_string())

    # By sample size
    if 'nmi_improvement' in df.columns:
        print("\n📈 BY SAMPLE SIZE")
        print("-" * 50)
        by_n = df.groupby('n_samples').agg({
            'nmi_dp': 'mean',
            'nmi_pyp': 'mean',
            'nmi_improvement': 'mean',
            'time_dp': 'mean',
            'time_pyp': 'mean'
        }).round(4)
        print(by_n.to_string())

    # Validation verdict
    print("\n" + "=" * 70)
    print("🔍 VALIDATION VERDICT")
    print("=" * 70)

    if 'nmi_improvement' in df.columns and df['nmi_improvement'].notna().any():
        mean_improvement = df['nmi_improvement'].mean()

        if mean_improvement >= 10:
            print(f"✅ REPORTS VALIDATED: PYP shows {mean_improvement:.1f}% average NMI improvement")
            print("   Recommendation: Use PYP for HIMARI clustering")
        elif mean_improvement >= 5:
            print(f"⚠️  PARTIAL VALIDATION: PYP shows {mean_improvement:.1f}% improvement")
            print("   Recommendation: PYP offers modest gains, weigh against complexity")
        else:
            print(f"❌ REPORTS NOT VALIDATED: Only {mean_improvement:.1f}% improvement observed")
            print("   Recommendation: Stick with DP (simpler, similar performance)")

        # K recovery check
        mean_k_error_dp = df['k_error_dp'].mean()
        mean_k_error_pyp = df['k_error_pyp'].mean()
        print(f"\n   K recovery - DP error: {mean_k_error_dp:.1f}, PYP error: {mean_k_error_pyp:.1f}")

        if mean_k_error_pyp < mean_k_error_dp * 0.7:
            print("   ✅ PYP recovers true K significantly better")
        else:
            print("   ⚠️  K recovery difference not substantial")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Validate Pitman-Yor vs Dirichlet Process clustering on power-law data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single comparison
  python validate_pyp_vs_dp.py --n_samples 50000 --n_features 60 --n_clusters 30

  # Full parameter sweep
  python validate_pyp_vs_dp.py --run_sweep

  # Quick test
  python validate_pyp_vs_dp.py --n_samples 10000 --quick
        """
    )

    parser.add_argument('--n_samples', type=int, default=50000,
                        help='Number of samples (default: 50000)')
    parser.add_argument('--n_features', type=int, default=60,
                        help='Number of features (default: 60)')
    parser.add_argument('--n_clusters', type=int, default=30,
                        help='True number of clusters (default: 30)')
    parser.add_argument('--zipf_exponent', type=float, default=1.5,
                        help='Zipf exponent for power-law (default: 1.5)')
    parser.add_argument('--discount', type=float, default=0.25,
                        help='PYP discount parameter (default: 0.25)')
    parser.add_argument('--concentration', type=float, default=1.0,
                        help='Concentration parameter (default: 1.0)')
    parser.add_argument('--max_components', type=int, default=100,
                        help='Maximum components for truncation (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--run_sweep', action='store_true',
                        help='Run full parameter sweep')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with fewer iterations')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file for sweep results')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')

    args = parser.parse_args()

    # Print system info
    print("=" * 70)
    print("PYP vs DP CLUSTERING VALIDATION")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__ if PYRO_AVAILABLE else 'N/A'}")
    print(f"Pyro:    {pyro.__version__ if PYRO_AVAILABLE else 'N/A'}")
    print(f"Device:  {GPU_NAME}")
    if GPU_AVAILABLE:
        print(f"GPU Memory: {GPU_MEMORY:.1f} GB")
    print("=" * 70)

    if args.run_sweep:
        # Parameter sweep
        if args.quick:
            zipf_exponents = [1.2, 2.0]
            sample_sizes = [10000, 50000]
            seeds = [42]
        else:
            zipf_exponents = [1.2, 1.5, 2.0, 2.5]
            sample_sizes = [10000, 50000, 100000]
            seeds = [42, 123, 456]

        df = run_parameter_sweep(
            zipf_exponents=zipf_exponents,
            sample_sizes=sample_sizes,
            n_features=args.n_features,
            n_true_clusters=args.n_clusters,
            seeds=seeds,
            verbose=not args.quiet
        )

        # Save results
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")

        # Print summary
        print_sweep_summary(df)

        # Print full table
        print("\n📋 FULL RESULTS TABLE")
        print("-" * 70)
        cols = ['zipf_exp', 'n_samples', 'k_dp', 'k_pyp', 'nmi_dp', 'nmi_pyp',
                'nmi_improvement', 'time_dp', 'time_pyp']
        cols = [c for c in cols if c in df.columns]
        print(df[cols].to_string(index=False))

    else:
        # Single comparison
        result = run_single_comparison(
            n_samples=args.n_samples,
            n_features=args.n_features,
            n_true_clusters=args.n_clusters,
            zipf_exponent=args.zipf_exponent,
            max_components=args.max_components,
            pyp_discount=args.discount,
            concentration=args.concentration,
            seed=args.seed,
            verbose=not args.quiet
        )

        # Print validation conclusion
        if result['pyp'] and result['pyp']['NMI']:
            nmi_improvement = (result['pyp']['NMI'] - result['dp']['NMI']) / result['dp']['NMI'] * 100

            print("\n" + "=" * 70)
            print("🔍 CONCLUSION")
            print("=" * 70)

            if nmi_improvement >= 10:
                print(f"✅ PYP shows significant improvement ({nmi_improvement:.1f}%)")
                print("   Reports VALIDATED for this configuration")
            elif nmi_improvement >= 5:
                print(f"⚠️  PYP shows modest improvement ({nmi_improvement:.1f}%)")
            else:
                print(f"❌ PYP shows minimal improvement ({nmi_improvement:.1f}%)")
                print("   Reports NOT validated for this configuration")


if __name__ == '__main__':
    main()
